import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from fraud_gnn.data.samld import compute_structural_features
from fraud_gnn.models.graphsage_samld import StructuralOnlyGraphSAGE, merge_rare_types, extract_embeddings
from fraud_gnn.evaluation.significance_tests import paired_significance_test
from fraud_gnn.utils.metrics import load_checkpoint

CONFIG_PATH = "configs/samld_binary.yaml"
TYPOLOGY_CONFIG_PATH = "configs/samld_typology.yaml"
BEHAVIOURAL_PATH = "samld_behavioural_features.pt"
SEEDS = (42, 1, 7, 123, 2024)
BEHAVIOURAL_CHANGE_1_ID = 0
BEHAVIOURAL_CHANGE_2_ID = 1


def fit_and_evaluate(X, y_type_remapped, train_mask, test_mask, remap, seed_label=None, verbose=True):
    illicit_train = train_mask & (y_type_remapped != -1)
    illicit_test = test_mask & (y_type_remapped != -1)

    X_train, y_train = X[illicit_train.numpy()], y_type_remapped[illicit_train].numpy()
    X_test, y_test = X[illicit_test.numpy()], y_type_remapped[illicit_test].numpy()

    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    bc1_new_id = remap.get(BEHAVIOURAL_CHANGE_1_ID)
    bc2_new_id = remap.get(BEHAVIOURAL_CHANGE_2_ID)
    labels_present = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    per_class = f1_score(y_test, y_pred, average=None, zero_division=0, labels=labels_present)
    label_to_f1 = dict(zip(labels_present, per_class))

    bc1_f1 = label_to_f1.get(bc1_new_id)
    bc2_f1 = label_to_f1.get(bc2_new_id)

    if verbose:
        print(f"  seed={seed_label}: macro_f1={macro_f1:.4f}  BC1_f1={bc1_f1}  BC2_f1={bc2_f1}")

    return {"macro_f1": macro_f1, "bc1_f1": bc1_f1, "bc2_f1": bc2_f1}


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(TYPOLOGY_CONFIG_PATH, encoding="utf-8") as f:
        typ_cfg = yaml.safe_load(f)

    print("Loading data + new behavioral features...")
    data = torch.load(f"{cfg['output_prefix']}.pt", map_location="cpu", weights_only=False)
    x = data["x"]
    edge_index = data["edge_index"]
    y_type = data["y_type"]
    train_mask = data["train_mask"]
    test_mask = data["test_mask"]

    behavioural = torch.load(BEHAVIOURAL_PATH, map_location="cpu", weights_only=False)
    behavioural_features = behavioural["features_scaled"].numpy()
    print(f"Behavioral features shape: {behavioural_features.shape}  names: {behavioural['feature_names']}")

    device = torch.device("cpu")
    remap, num_new_types, _ = merge_rare_types(
        y_type, train_mask, min_train_samples=typ_cfg["min_train_samples_per_type"],
    )
    y_type_remapped = y_type.clone()
    for old_id, new_id in remap.items():
        y_type_remapped[y_type == old_id] = new_id

    struct_raw, _ = compute_structural_features(edge_index, x.shape[0])
    illicit_train_np = (train_mask & (y_type_remapped != -1)).numpy()
    struct_scaler = StandardScaler().fit(struct_raw[illicit_train_np])
    struct_scaled = struct_scaler.transform(struct_raw)

    baseline_metrics, enhanced_metrics = [], []
    for seed in SEEDS:
        print(f"\nSeed {seed}:")
        model = StructuralOnlyGraphSAGE(in_channels=x.shape[1], hidden_channels=cfg["hidden_channels"],
                                         out_channels=cfg["out_channels"]).to(device)
        load_checkpoint(model, f"{cfg['checkpoint_dir']}/samld_seed_{seed}.pt", map_location=device)
        h2 = extract_embeddings(model, x, edge_index, device).numpy()

        X_baseline = np.concatenate([h2, struct_scaled], axis=1)
        X_enhanced = np.concatenate([h2, struct_scaled, behavioural_features], axis=1)

        print(" baseline (embedding + structural, no behavioral):")
        m_base = fit_and_evaluate(X_baseline, y_type_remapped, train_mask, test_mask, remap, seed_label=seed)
        baseline_metrics.append(m_base)

        print(" enhanced (+ behavioral features):")
        m_enh = fit_and_evaluate(X_enhanced, y_type_remapped, train_mask, test_mask, remap, seed_label=seed)
        enhanced_metrics.append(m_enh)

    print(f"\n\n{'=' * 70}\n=== Summary across {len(SEEDS)} seeds ===\n{'=' * 70}")
    for label, metrics in [("BASELINE", baseline_metrics), ("ENHANCED (+behavioral)", enhanced_metrics)]:
        macro = np.array([m["macro_f1"] for m in metrics])
        bc1 = np.array([m["bc1_f1"] for m in metrics if m["bc1_f1"] is not None])
        bc2 = np.array([m["bc2_f1"] for m in metrics if m["bc2_f1"] is not None])
        print(f"\n{label}:")
        print(f"  macro_f1: {macro.mean():.4f} +/- {macro.std(ddof=1):.4f}")
        if len(bc1):
            print(f"  Behavioural_Change_1 F1: {bc1.mean():.4f} +/- {bc1.std(ddof=1):.4f}")
        else:
            print("  Behavioural_Change_1: not present as its own class this run")
        if len(bc2):
            print(f"  Behavioural_Change_2 F1: {bc2.mean():.4f} +/- {bc2.std(ddof=1):.4f}")
        else:
            print("  Behavioural_Change_2: not present as its own class this run")

    print(f"\n\n{'=' * 70}\n=== Paired significance test: baseline vs enhanced ({len(SEEDS)} seeds) ===\n{'=' * 70}")
    macro_base = [m["macro_f1"] for m in baseline_metrics]
    macro_enh = [m["macro_f1"] for m in enhanced_metrics]
    paired_significance_test(macro_enh, macro_base, "Enhanced (macro-F1)", "Baseline (macro-F1)")

    for label, key in [("BC1", "bc1_f1"), ("BC2", "bc2_f1")]:
        pairs = [(b[key], e[key]) for b, e in zip(baseline_metrics, enhanced_metrics)
                 if b[key] is not None and e[key] is not None]
        if len(pairs) >= 2:
            base_vals, enh_vals = zip(*pairs)
            paired_significance_test(list(enh_vals), list(base_vals), f"Enhanced ({label} F1)", f"Baseline ({label} F1)")
        else:
            print(f"\nNot enough seeds where {label} appeared in both baseline and enhanced to run a paired test.")


if __name__ == "__main__":
    main()
