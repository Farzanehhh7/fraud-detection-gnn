import os
import sys
import io
import datetime

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False


RUN_LOGS_DIR = "run-logs"
os.makedirs(RUN_LOGS_DIR, exist_ok=True)
_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_PATH = os.path.join(RUN_LOGS_DIR, f"samld_15_{_timestamp}.log")
_log_file = open(RUN_LOG_PATH, "w", encoding="utf-8")

_stdout_original = sys.stdout
_stderr_original = sys.stderr
sys.stdout = _Tee(_stdout_original, _log_file)
sys.stderr = _Tee(_stderr_original, _log_file)

print(f"[این اجرا هم‌زمان توی این فایل ذخیره می‌شه: {RUN_LOG_PATH}]")

import atexit


def _cleanup_logging():
    sys.stdout = _stdout_original
    sys.stderr = _stderr_original
    try:
        _log_file.close()
    except Exception:
        pass


atexit.register(_cleanup_logging)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from xgboost import XGBClassifier

from fraud_gnn.models.archived.hybrid_and_skipgcn import EmbeddingExtractor
from fraud_gnn.evaluation.significance_tests import paired_significance_test
from fraud_gnn.utils.metrics import evaluate_binary, find_best_threshold
from fraud_gnn.utils.run_logger import log_run

CONFIG_PATH = "configs/samld_binary.yaml"
METRICS = ("AUC", "PR-AUC", "F1", "Precision", "Recall", "MCC")
METRICS_DIR = "outputs/metrics"
LOGS_DIR = "outputs/logs"
GRAPHSAGE_REFERENCE_CSV = f"{METRICS_DIR}/samld_graphsage_binary_15seed_results.csv"
EMBEDDING_DIM = 64


def safe_log_run(**kwargs):
    try:
        log_run(**kwargs)
    except Exception as exc:
        print(f"[هشدار] log_run شکست خورد (seed={kwargs.get('seed')}): {exc}")


def train_eval_hybrid(seed, x, edge_index, y, train_mask, val_mask, test_mask,
                       hidden_channels, embedding_dim=EMBEDDING_DIM, device="cpu"):
    torch.manual_seed(seed)
    emb_model = EmbeddingExtractor(
        in_channels=x.shape[1], hidden_channels=hidden_channels, embedding_dim=embedding_dim,
    ).to(device)
    optimizer = torch.optim.Adam(emb_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    emb_model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = emb_model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

    emb_model.eval()
    with torch.no_grad():
        graph_embeddings = emb_model(x, edge_index).cpu().numpy()

    x_raw = x.cpu().numpy()
    X_combined = np.hstack([x_raw, graph_embeddings])
    y_np = y.cpu().numpy()

    train_mask_np = train_mask.cpu().numpy()
    val_mask_np = val_mask.cpu().numpy()
    test_mask_np = test_mask.cpu().numpy()

    X_train, y_train = X_combined[train_mask_np], y_np[train_mask_np]
    X_val, y_val = X_combined[val_mask_np], y_np[val_mask_np]
    X_test, y_test = X_combined[test_mask_np], y_np[test_mask_np]

    n_neg_train, n_pos_train = (y_train == 0).sum(), (y_train == 1).sum()
    xgb_hybrid = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=float(n_neg_train / n_pos_train), eval_metric="logloss",
        random_state=seed,
    )
    xgb_hybrid.fit(X_train, y_train)

    probs_val = xgb_hybrid.predict_proba(X_val)[:, 1]
    best_t, _ = find_best_threshold(y_val, probs_val)

    probs_test = xgb_hybrid.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= best_t).astype(int)

    metrics = evaluate_binary("SAML-D Hybrid (GraphSAGE+XGBoost)", y_test, preds_test, probs_test, verbose=False)
    metrics["threshold"] = best_t
    return metrics


def load_graphsage_reference(seeds):
    df = pd.read_csv(GRAPHSAGE_REFERENCE_CSV)
    return df.set_index("seed").loc[list(seeds)].reset_index()


def compare_to_graphsage(df_sage, df_hybrid, seeds, summary_rows):
    df_sage = df_sage.sort_values("seed").reset_index(drop=True)
    df_hybrid = df_hybrid.sort_values("seed").reset_index(drop=True)

    if not (df_sage["seed"].values == df_hybrid["seed"].values).all():
        raise ValueError("seed mismatch بین GraphSAGE و Hybrid — مقایسه‌ی element-wise معتبر نیست.")

    print(f"\n{'=' * 70}\nمقایسه‌ی آماری: Hybrid در برابر GraphSAGE (۱۵ seed، SAML-D)\n{'=' * 70}")
    for metric in METRICS:
        t_stat, p_value = paired_significance_test(
            df_sage[metric].tolist(), df_hybrid[metric].tolist(),
            f"GraphSAGE ({metric})", f"Hybrid ({metric})",
        )
        wins_sage = int((df_sage[metric].values > df_hybrid[metric].values).sum())
        n = len(df_sage)
        summary_rows.append({
            "architecture": "Hybrid",
            "metric": metric,
            "mean_graphsage": df_sage[metric].mean(),
            "mean_variant": df_hybrid[metric].mean(),
            "mean_diff": (df_sage[metric] - df_hybrid[metric]).mean(),
            "t_stat": t_stat,
            "p_value": p_value,
            "significant_at_0.05": bool(p_value < 0.05),
            "graphsage_win_ratio": f"{wins_sage}/{n}",
        })


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seeds = cfg["seeds"]

    print("Loading real SAML-D v3 data...")
    data_dict = torch.load(f"{cfg['output_prefix']}.pt", map_location="cpu", weights_only=False)
    x = data_dict["x"].to(device)
    edge_index = data_dict["edge_index"].to(device)
    y = data_dict["y_binary"].to(device)
    train_mask = data_dict["train_mask"].to(device)
    val_mask = data_dict["val_mask"].to(device)
    test_mask = data_dict["test_mask"].to(device)

    if not os.path.exists(GRAPHSAGE_REFERENCE_CSV):
        raise FileNotFoundError(
            f"فایل مرجع GraphSAGE پیدا نشد: {GRAPHSAGE_REFERENCE_CSV} — "
            "اول 14_significance_test_binary_15seed.py رو اجرا کن."
        )

    rows = []
    for seed in seeds:
        metrics = train_eval_hybrid(
            seed, x, edge_index, y, train_mask, val_mask, test_mask,
            hidden_channels=cfg["hidden_channels"], device=device,
        )
        metrics["seed"] = seed
        rows.append(metrics)
        print(f"seed={seed:5d}  F1={metrics['F1']:.4f}  AUC={metrics['AUC']:.4f}  "
              f"threshold={metrics['threshold']:.2f}")

        safe_log_run(phase="samld", experiment="15_hybrid_binary_15seed", seed=seed,
                     metric_name="F1", metric_value=metrics["F1"],
                     checkpoint_path="", log_path=f"{LOGS_DIR}/samld_15.log")

    df_hybrid = pd.DataFrame(rows)
    df_hybrid.to_csv(f"{METRICS_DIR}/samld_hybrid_15seed_results.csv", index=False)
    print(f"\nذخیره شد: {METRICS_DIR}/samld_hybrid_15seed_results.csv")

    df_sage = load_graphsage_reference(seeds)

    summary_rows = []
    compare_to_graphsage(df_sage, df_hybrid, seeds, summary_rows)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(f"{METRICS_DIR}/samld_hybrid_vs_graphsage_15seed_summary.csv", index=False)

    print(f"\n{'=' * 70}\nجدول نهایی\n{'=' * 70}")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(df_summary.round(4).to_string(index=False))

    return df_summary


if __name__ == "__main__":
    main()
