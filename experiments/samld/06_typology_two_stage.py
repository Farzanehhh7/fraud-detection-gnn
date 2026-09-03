import numpy as np
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from fraud_gnn.data.samld import compute_structural_features
from fraud_gnn.models.graphsage_samld import (
    StructuralOnlyGraphSAGE, merge_rare_types, extract_embeddings,
    load_type_names, fit_and_evaluate_type_classifier,
)
from fraud_gnn.utils.metrics import load_checkpoint
from fraud_gnn.utils.run_logger import log_run

CONFIG_PATH = "configs/samld_binary.yaml"
TYPOLOGY_CONFIG_PATH = "configs/samld_typology.yaml"


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(TYPOLOGY_CONFIG_PATH, encoding="utf-8") as f:
        typ_cfg = yaml.safe_load(f)

    print("Loading real SAML-D v3 data...")
    data_dict = torch.load(f"{cfg['output_prefix']}.pt", map_location="cpu", weights_only=False)
    x = data_dict["x"]
    edge_index = data_dict["edge_index"]
    y_type = data_dict["y_type"]
    train_mask = data_dict["train_mask"]
    test_mask = data_dict["test_mask"]
    num_nodes = x.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    remap, num_new_types, _ = merge_rare_types(
        y_type, train_mask, min_train_samples=typ_cfg["min_train_samples_per_type"],
    )
    y_type_remapped = y_type.clone()
    for old_id, new_id in remap.items():
        y_type_remapped[y_type == old_id] = new_id

    type_names = load_type_names(typ_cfg["type_names_path"])

    struct_features_raw, struct_feature_names = compute_structural_features(edge_index, num_nodes)
    illicit_train_np = (train_mask & (y_type_remapped != -1)).numpy()
    scaler = StandardScaler().fit(struct_features_raw[illicit_train_np])
    struct_features_scaled = scaler.transform(struct_features_raw)

    baseline_metrics, enhanced_metrics = [], []

    for seed in cfg["seeds"]:
        ckpt_path = f"{cfg['checkpoint_dir']}/samld_seed_{seed}.pt"
        print(f"\n{'=' * 60}\nSeed {seed}: loading {ckpt_path}\n{'=' * 60}")

        model = StructuralOnlyGraphSAGE(
            in_channels=x.shape[1], hidden_channels=cfg["hidden_channels"], out_channels=cfg["out_channels"],
        ).to(device)
        load_checkpoint(model, ckpt_path, map_location=device)

        h2 = extract_embeddings(model, x, edge_index, device).numpy()

        m_base, _ = fit_and_evaluate_type_classifier(
            torch.tensor(h2), y_type_remapped, train_mask, test_mask, num_new_types,
            seed_label=seed, verbose=False, type_names=type_names,
        )
        baseline_metrics.append(m_base)
        log_run(phase="samld", experiment="06_typology_two_stage", seed=seed,
                metric_name="macro_f1_embedding_only", metric_value=m_base["macro_f1"],
                checkpoint_path=ckpt_path, log_path="outputs/logs/samld_06.log")

        x_combined = np.concatenate([h2, struct_features_scaled], axis=1)
        m_enh, _ = fit_and_evaluate_type_classifier(
            torch.tensor(x_combined), y_type_remapped, train_mask, test_mask, num_new_types,
            seed_label=seed, verbose=False, type_names=type_names,
        )
        enhanced_metrics.append(m_enh)
        log_run(phase="samld", experiment="06_typology_two_stage", seed=seed,
                metric_name="macro_f1_embedding_structural", metric_value=m_enh["macro_f1"],
                checkpoint_path=ckpt_path, log_path="outputs/logs/samld_06.log")

    for label, metrics_list in [("BASELINE (embedding only)", baseline_metrics),
                                 ("ENHANCED (embedding + structural)", enhanced_metrics)]:
        vals = np.array([m["macro_f1"] for m in metrics_list])
        print(f"\n{label}: macro_f1 = {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")

    return baseline_metrics, enhanced_metrics


if __name__ == "__main__":
    main()
