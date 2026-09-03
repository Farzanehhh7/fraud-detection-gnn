import numpy as np
import torch
import yaml
from sklearn.metrics import precision_score, recall_score

from fraud_gnn.models.graphsage_samld import StructuralOnlyGraphSAGE
from fraud_gnn.utils.metrics import load_checkpoint
from fraud_gnn.utils.run_logger import log_run

CONFIG_PATH = "configs/samld_binary.yaml"
PERCENTILES = (90, 95, 99)


def report_at_percentile_thresholds_capture(y_true, y_prob, percentiles):
    results = {}
    for p in percentiles:
        threshold = np.percentile(y_prob, p)
        preds = (y_prob >= threshold).astype(int)
        flagged = preds.sum()
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        results[p] = {"threshold": threshold, "flagged": flagged, "precision": precision, "recall": recall}
        print(f"  صدک {p:5.1f} | threshold={threshold:.4f} | flagged={flagged:6d} | "
              f"Precision={precision:.4f} | Recall={recall:.4f}")
    return results


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("Loading real SAML-D v3 data...")
    data_dict = torch.load(f"{cfg['output_prefix']}.pt", map_location="cpu", weights_only=False)
    x = data_dict["x"]
    edge_index = data_dict["edge_index"]
    y_binary = data_dict["y_binary"]
    test_mask = data_dict["test_mask"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_seed_results = []
    for seed in cfg["seeds"]:
        print(f"\n{'=' * 60}\nSeed {seed} (reusing existing checkpoint, no retraining)\n{'=' * 60}")
        model = StructuralOnlyGraphSAGE(
            in_channels=x.shape[1], hidden_channels=cfg["hidden_channels"], out_channels=cfg["out_channels"],
        ).to(device)
        load_checkpoint(model, f"{cfg['checkpoint_dir']}/samld_seed_{seed}.pt", map_location=device)

        model.eval()
        with torch.no_grad():
            out = model(x.to(device), edge_index.to(device))
            probs_all = torch.softmax(out, dim=1)[:, 1].cpu().numpy()

        y_test = y_binary[test_mask].numpy()
        probs_test = probs_all[test_mask.numpy()]

        seed_results = report_at_percentile_thresholds_capture(y_test, probs_test, PERCENTILES)
        all_seed_results.append(seed_results)

        for p in PERCENTILES:
            log_run(
                phase="samld", experiment="07_fair_reference_baseline", seed=seed,
                metric_name=f"precision_p{p}", metric_value=seed_results[p]["precision"],
                checkpoint_path=f"{cfg['checkpoint_dir']}/samld_seed_{seed}.pt",
                log_path="outputs/logs/samld_07.log",
                notes=f"recall_p{p}={seed_results[p]['recall']:.4f}",
            )

    n_seeds = len(cfg["seeds"])
    print(f"\n\n{'=' * 60}\n=== FAIR {n_seeds}-SEED BINARY-ONLY REFERENCE (percentile-based) ===\n{'=' * 60}")
    for p in PERCENTILES:
        precisions = np.array([r[p]["precision"] for r in all_seed_results])
        recalls = np.array([r[p]["recall"] for r in all_seed_results])
        print(f"صدک {p:5.1f}:  Precision = {precisions.mean():.4f} +/- {precisions.std(ddof=1):.4f}   "
              f"Recall = {recalls.mean():.4f} +/- {recalls.std(ddof=1):.4f}")

    return all_seed_results


if __name__ == "__main__":
    main()
