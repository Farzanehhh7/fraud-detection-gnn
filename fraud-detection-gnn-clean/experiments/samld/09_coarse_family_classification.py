import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from fraud_gnn.data.samld import compute_structural_features, build_family_labels
from fraud_gnn.models.graphsage_samld import StructuralOnlyGraphSAGE, extract_embeddings
from fraud_gnn.utils.metrics import load_checkpoint
from fraud_gnn.utils.run_logger import log_run

CONFIG_PATH = "configs/samld_binary.yaml"
TYPOLOGY_CONFIG_PATH = "configs/samld_typology.yaml"


def fit_and_evaluate(X, y_family, train_mask, test_mask, family_names, seed_label=None, verbose=True):
    illicit_train = train_mask & (y_family != -1)
    illicit_test = test_mask & (y_family != -1)

    X_train, y_train = X[illicit_train.numpy()], y_family[illicit_train].numpy()
    X_test, y_test = X[illicit_test.numpy()], y_family[illicit_test].numpy()

    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    if verbose:
        labels_present = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
        target_names = [family_names[i] for i in labels_present]
        print(f"\n--- seed={seed_label} ---")
        print(classification_report(y_test, y_pred, labels=labels_present,
                                     target_names=target_names, zero_division=0))

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(TYPOLOGY_CONFIG_PATH, encoding="utf-8") as f:
        typ_cfg = yaml.safe_load(f)

    family_map = {int(k): v for k, v in typ_cfg["family_map"].items()}
    family_names = {int(k): v for k, v in typ_cfg["family_names"].items()}

    print("Loading real SAML-D v3 data...")
    data_dict = torch.load(f"{cfg['output_prefix']}.pt", map_location="cpu", weights_only=False)
    x = data_dict["x"]
    edge_index = data_dict["edge_index"]
    y_type = data_dict["y_type"]
    train_mask = data_dict["train_mask"]
    test_mask = data_dict["test_mask"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_family = build_family_labels(y_type, family_map)
    print("\nFamily sample counts (train):")
    for fam_id, fam_name in family_names.items():
        n = ((y_family == fam_id) & train_mask).sum().item()
        print(f"  {fam_name:25s}: {n}")

    struct_features_raw, _ = compute_structural_features(edge_index, x.shape[0])
    illicit_train_np = (train_mask & (y_family != -1)).numpy()
    scaler = StandardScaler().fit(struct_features_raw[illicit_train_np])
    struct_features_scaled = scaler.transform(struct_features_raw)

    all_metrics = []
    for seed in cfg["seeds"]:
        print(f"\n{'=' * 60}\nSeed {seed}\n{'=' * 60}")
        model = StructuralOnlyGraphSAGE(
            in_channels=x.shape[1], hidden_channels=cfg["hidden_channels"], out_channels=cfg["out_channels"],
        ).to(device)
        ckpt_path = f"{cfg['checkpoint_dir']}/samld_seed_{seed}.pt"
        load_checkpoint(model, ckpt_path, map_location=device)

        h2 = extract_embeddings(model, x, edge_index, device).numpy()
        X = np.concatenate([h2, struct_features_scaled], axis=1)

        m = fit_and_evaluate(X, y_family, train_mask, test_mask, family_names, seed_label=seed, verbose=True)
        all_metrics.append(m)

        log_run(phase="samld", experiment="09_coarse_family_classification", seed=seed,
                metric_name="macro_f1", metric_value=m["macro_f1"],
                checkpoint_path=ckpt_path, log_path="outputs/logs/samld_09.log")

    print(f"\n\n{'=' * 60}\n=== Coarse family classifier — summary across {len(cfg['seeds'])} seeds ===\n{'=' * 60}")
    for key in ("accuracy", "macro_f1", "weighted_f1"):
        vals = np.array([m[key] for m in all_metrics])
        print(f"  {key:12s}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")

    return all_metrics


if __name__ == "__main__":
    main()
