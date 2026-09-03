import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score

from fraud_gnn.data.elliptic import load_elliptic
from fraud_gnn.models.graphsage_elliptic import StructuralOnlyGraphSAGE
from fraud_gnn.utils.metrics import evaluate_binary, find_best_threshold, run_multi_seed, EarlyStopper, log_experiment

CONFIG_PATH = "configs/elliptic_binary.yaml"
MAX_EPOCHS = 200
PATIENCE = 20
LOG_PATH = "hparam_search_log.csv"
SEARCH_SEEDS = (42, 1, 7)
FINAL_SEEDS = (42, 1, 7, 123, 2024)


def run_one_seed(seed, lr, hidden_channels, x, edge_index, y, train_mask, val_mask, test_mask, class_weights):
    model = StructuralOnlyGraphSAGE(in_channels=x.shape[1], hidden_channels=hidden_channels, out_channels=2).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    stopper = EarlyStopper(patience=PATIENCE, mode="max")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(x, edge_index)
            val_probs = F.softmax(val_out, dim=1)[val_mask, 1].cpu().numpy()
        val_auprc = average_precision_score(y[val_mask].cpu().numpy(), val_probs)

        if stopper.step(val_auprc, model, epoch=epoch):
            break

    stopper.restore_best(model)
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

    metrics = evaluate_binary("SAGE", y_test, preds_test, probs_test, verbose=False)
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

    search_space = {"lr": [0.001, 0.005, 0.01], "hidden_channels": [32, 64, 128]}

    results_summary = []
    print("\n=== مرحله جست‌وجو ===")
    for lr in search_space["lr"]:
        for hc in search_space["hidden_channels"]:
            name = f"lr={lr}, hidden={hc}"
            df, summary = run_multi_seed(
                lambda seed, lr=lr, hc=hc: run_one_seed(
                    seed, lr, hc, x, edge_index, y, train_mask, val_mask, test_mask, class_weights,
                ),
                seeds=SEARCH_SEEDS, name=name, verbose=False,
            )
            mean_f1 = summary.loc["mean", "F1"]
            std_f1 = summary.loc["std", "F1"]
            print(f"{name:25s} F1 = {mean_f1:.4f} ± {std_f1:.4f}")
            log_experiment(LOG_PATH, {"lr": lr, "hidden_channels": hc, "F1_mean": mean_f1, "F1_std": std_f1})
            results_summary.append((lr, hc, mean_f1, std_f1))

    best_lr, best_hc, best_f1, _ = max(results_summary, key=lambda r: r[2])
    print(f"\nبهترین پیکربندی مرحله جست‌وجو: lr={best_lr}, hidden_channels={best_hc}, F1={best_f1:.4f}")

    print("\n\n=== تایید نهایی با پنج seed کامل ===")
    df_final, summary_final = run_multi_seed(
        lambda seed: run_one_seed(
            seed, best_lr, best_hc, x, edge_index, y, train_mask, val_mask, test_mask, class_weights,
        ),
        seeds=FINAL_SEEDS, name="بهترین پیکربندی، تایید نهایی",
    )

    print(f"\n{'بهترین پیکربندی، پنج seed کامل':32s} F1 = {summary_final.loc['mean', 'F1']:.4f} ± {summary_final.loc['std', 'F1']:.4f}")
    print(f"{'مرجع lr=0.005 hidden=64، فاز یک':32s} F1 = 0.4427 ± 0.0323")

    return best_lr, best_hc, df_final, summary_final


if __name__ == "__main__":
    main()
