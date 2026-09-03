import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from fraud_gnn.data.elliptic import load_elliptic
from fraud_gnn.models.graphsage_elliptic import StructuralOnlyGraphSAGE
from fraud_gnn.models.baselines import TIEBaseline
from fraud_gnn.evaluation.significance_tests import paired_significance_test
from fraud_gnn.utils.metrics import evaluate_binary, find_best_threshold, run_multi_seed
from fraud_gnn.utils.run_logger import log_run

CONFIG_PATH = "configs/elliptic_binary.yaml"
EPOCHS = 200
TEMPORAL_DIM = 32


def train_eval_graphsage(seed, x, edge_index, y, train_mask, val_mask, test_mask, class_weights, hidden_channels):
    model = StructuralOnlyGraphSAGE(in_channels=x.shape[1], hidden_channels=hidden_channels, out_channels=2).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

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

    metrics = evaluate_binary("GraphSAGE", y_test, preds_test, probs_test, verbose=False)
    metrics["threshold"] = best_t
    return metrics


def train_eval_tie(seed, x, y, time_steps, num_timesteps, train_mask, val_mask, test_mask, class_weights, hidden_channels):
    model = TIEBaseline(
        in_channels=x.shape[1], num_timesteps=num_timesteps, temporal_dim=TEMPORAL_DIM,
        hidden_channels=hidden_channels, out_channels=2,
    ).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, time_steps)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(x, time_steps)
        probs_all = F.softmax(out, dim=1)[:, 1].cpu().numpy()

    y_val = y[val_mask].cpu().numpy()
    probs_val = probs_all[val_mask.cpu().numpy()]
    best_t, _ = find_best_threshold(y_val, probs_val)

    y_test = y[test_mask].cpu().numpy()
    probs_test = probs_all[test_mask.cpu().numpy()]
    preds_test = (probs_test >= best_t).astype(int)

    metrics = evaluate_binary("TIE Baseline", y_test, preds_test, probs_test, verbose=False)
    metrics["threshold"] = best_t
    return metrics


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

    time_steps = (data["time_steps_raw"] - 1).to(device)
    num_timesteps = int(time_steps.max().item()) + 1

    seeds = cfg["seeds"]

    print(f"\n=== GraphSAGE, {len(seeds)} seeds ===")
    df_sage, summary_sage = run_multi_seed(
        lambda seed: train_eval_graphsage(seed, x, edge_index, y, train_mask, val_mask, test_mask,
                                           class_weights, cfg["hidden_channels"]),
        seeds=seeds, name="GraphSAGE ساختاری",
    )

    print(f"\n=== TIE Baseline, {len(seeds)} seeds ===")
    df_tie, summary_tie = run_multi_seed(
        lambda seed: train_eval_tie(seed, x, y, time_steps, num_timesteps, train_mask, val_mask, test_mask,
                                     class_weights, cfg["hidden_channels"]),
        seeds=seeds, name="TIE Baseline",
    )

    for metric in ("F1", "PR-AUC", "AUC"):
        print(f"\n=== آزمون معناداری روی {metric}، {len(seeds)} seed ===")
        paired_significance_test(
            df_sage[metric].tolist(), df_tie[metric].tolist(),
            f"GraphSAGE ساختاری ({metric})", f"TIE بدون گراف ({metric})",
        )
        wins_sage = int((df_sage[metric].values > df_tie[metric].values).sum())
        print(f"نسبت برد به‌ازای {len(seeds)} seed: GraphSAGE {wins_sage}/{len(seeds)}   "
              f"TIE {len(seeds) - wins_sage}/{len(seeds)}")

    for seed, row in zip(seeds, df_sage.to_dict("records")):
        log_run(phase="elliptic", experiment="07_significance_test_15seed", seed=seed,
                metric_name="F1_graphsage", metric_value=row["F1"],
                checkpoint_path="", log_path="outputs/logs/elliptic_07.log")
    for seed, row in zip(seeds, df_tie.to_dict("records")):
        log_run(phase="elliptic", experiment="07_significance_test_15seed", seed=seed,
                metric_name="F1_tie", metric_value=row["F1"],
                checkpoint_path="", log_path="outputs/logs/elliptic_07.log")

    df_sage.to_csv("outputs/metrics/step32_graphsage_15seed_results.csv", index=False)
    df_tie.to_csv("outputs/metrics/step32_tie_15seed_results.csv", index=False)

    return df_sage, df_tie, summary_sage, summary_tie


if __name__ == "__main__":
    main()
