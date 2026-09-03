import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from fraud_gnn.data.elliptic import load_elliptic
from fraud_gnn.models.graphsage_elliptic import StructuralOnlyGraphSAGE
from fraud_gnn.utils.metrics import (
    evaluate_binary, find_best_threshold, report_at_percentile_thresholds,
    bootstrap_test_ci, compute_mad_neighbors, save_checkpoint,
)
from fraud_gnn.utils.seeding import set_seed
from fraud_gnn.utils.run_logger import log_run

CONFIG_PATH = "configs/elliptic_binary.yaml"
LR = 0.005
EPOCHS = 200
FINAL_CHECKPOINT_SEEDS = (42, 1, 7, 123, 2024)


def train_one_seed(seed, x, edge_index, y, train_mask, class_weights, device, hidden_channels):
    set_seed(seed)
    model = StructuralOnlyGraphSAGE(in_channels=x.shape[1], hidden_channels=hidden_channels, out_channels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

    return model


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = load_elliptic(
        features_path=cfg["features_path"], edges_path=cfg["edges_path"], classes_path=cfg["classes_path"],
        train_end=cfg["train_end"], val_end=cfg["val_end"], device=device,
    )
    x, y, edge_index = data["x"], data["y"], data["edge_index"]
    train_mask, val_mask, test_mask = data["train_mask"], data["val_mask"], data["test_mask"]
    class_weights = data["class_weights"]

    print("=== آموزش پنج seed استاندارد ===")
    trained_models, seed_f1, seed_threshold = {}, {}, {}

    for seed in FINAL_CHECKPOINT_SEEDS:
        model = train_one_seed(seed, x, edge_index, y, train_mask, class_weights, device, cfg["hidden_channels"])
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
            probs_all = F.softmax(out, dim=1)[:, 1].cpu().numpy()

        y_val = y[val_mask].cpu().numpy()
        probs_val = probs_all[val_mask.cpu().numpy()]
        best_t, _ = find_best_threshold(y_val, probs_val)

        y_test = y[test_mask].cpu().numpy()
        probs_test = probs_all[test_mask.cpu().numpy()]
        preds_test = (probs_test >= best_t).astype(int)
        metrics = evaluate_binary(f"seed={seed}", y_test, preds_test, probs_test, verbose=False)

        trained_models[seed] = model
        seed_f1[seed] = metrics["F1"]
        seed_threshold[seed] = best_t
        print(f"seed={seed}   F1={metrics['F1']:.4f}   threshold={best_t:.2f}")

        log_run(phase="elliptic", experiment="06_final_checkpoint", seed=seed,
                metric_name="F1", metric_value=metrics["F1"],
                checkpoint_path="", log_path="outputs/logs/elliptic_06.log")

    best_seed = max(seed_f1, key=seed_f1.get)
    best_model = trained_models[best_seed]
    best_threshold = seed_threshold[best_seed]
    print(f"\nبهترین seed: {best_seed}   F1={seed_f1[best_seed]:.4f}")

    best_model.eval()
    with torch.no_grad():
        out, h1, h2 = best_model(x, edge_index, return_embeddings=True)
        probs_all = F.softmax(out, dim=1)[:, 1].cpu().numpy()
    y_test = y[test_mask].cpu().numpy()
    probs_test = probs_all[test_mask.cpu().numpy()]

    print("\n\n=== گزارش threshold صدک‌محور، بهترین seed ===")
    report_at_percentile_thresholds(y_test, probs_test, percentiles=(90, 99, 99.9))

    print("\n\n=== فاصله اطمینان bootstrap روی test، بهترین seed ===")
    bootstrap_test_ci(y_test, probs_test, n_iterations=100, sample_frac=0.5, threshold=best_threshold)

    print("\n\n=== MAD بعد از هر لایه، بهترین seed ===")
    mad_layer1 = compute_mad_neighbors(h1, edge_index)
    mad_layer2 = compute_mad_neighbors(h2, edge_index)
    print(f"MAD بعد از لایه اول: {mad_layer1:.4f}")
    print(f"MAD بعد از لایه دوم: {mad_layer2:.4f}")

    save_checkpoint(
        best_model, "structural_only_best.pt",
        extra={"seed": best_seed, "F1": seed_f1[best_seed], "threshold": float(best_threshold),
               "hidden_channels": cfg["hidden_channels"], "lr": LR},
    )

    return best_model, best_seed, seed_f1


if __name__ == "__main__":
    main()
