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
RUN_LOG_PATH = os.path.join(RUN_LOGS_DIR, f"elliptic_11_{_timestamp}.log")
_log_file = open(RUN_LOG_PATH, "w", encoding="utf-8")

sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)

print(f"[این اجرا هم‌زمان توی این فایل ذخیره می‌شه: {RUN_LOG_PATH}]")

import atexit
atexit.register(_log_file.close)

import numpy as np
import pandas as pd
import torch
import yaml
from xgboost import XGBClassifier

from fraud_gnn.data.elliptic import load_elliptic
from fraud_gnn.evaluation.significance_tests import paired_significance_test
from fraud_gnn.utils.metrics import evaluate_binary, find_best_threshold, run_multi_seed
from fraud_gnn.utils.run_logger import log_run

CONFIG_PATH = "configs/elliptic_binary.yaml"
GRAPHSAGE_REFERENCE_CSV = "outputs/metrics/step32_graphsage_15seed_results.csv"
HYBRID_REFERENCE_CSV = "outputs/metrics/elliptic_hybrid_15seed_results.csv"
METRICS = ("AUC", "PR-AUC", "F1", "Precision", "Recall", "MCC")
METRICS_DIR = "outputs/metrics"
LOGS_DIR = "outputs/logs"


def safe_log_run(**kwargs):
    try:
        log_run(**kwargs)
    except Exception as exc:
        print(f"[هشدار] log_run شکست خورد (seed={kwargs.get('seed')}): {exc}")


def train_eval_xgb_raw(seed, x, y, train_mask, val_mask, test_mask):

    x_raw = x.cpu().numpy()
    y_np = y.cpu().numpy()

    train_mask_np = train_mask.cpu().numpy()
    val_mask_np = val_mask.cpu().numpy()
    test_mask_np = test_mask.cpu().numpy()

    X_train, y_train = x_raw[train_mask_np], y_np[train_mask_np]
    X_val, y_val = x_raw[val_mask_np], y_np[val_mask_np]
    X_test, y_test = x_raw[test_mask_np], y_np[test_mask_np]

    n_neg_train, n_pos_train = (y_train == 0).sum(), (y_train == 1).sum()
    xgb_raw = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=float(n_neg_train / n_pos_train), eval_metric="logloss",
        random_state=seed,
    )
    xgb_raw.fit(X_train, y_train)

    probs_val = xgb_raw.predict_proba(X_val)[:, 1]
    best_t, _ = find_best_threshold(y_val, probs_val)

    probs_test = xgb_raw.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= best_t).astype(int)

    metrics = evaluate_binary("XGBoost (raw features only)", y_test, preds_test, probs_test, verbose=False)
    metrics["threshold"] = best_t
    return metrics


def load_reference(path, seeds, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"فایل مرجع {label} پیدا نشد: {path}")
    df = pd.read_csv(path)
    return df.set_index("seed").loc[list(seeds)].reset_index()


def compare(df_a, df_b, label_a, label_b, summary_rows, comparison_name):
    df_a = df_a.sort_values("seed").reset_index(drop=True)
    df_b = df_b.sort_values("seed").reset_index(drop=True)

    if not (df_a["seed"].values == df_b["seed"].values).all():
        raise ValueError(f"seed mismatch بین {label_a} و {label_b} — مقایسه‌ی element-wise معتبر نیست.")

    print(f"\n{'=' * 70}\n{comparison_name}\n{'=' * 70}")
    for metric in METRICS:
        t_stat, p_value = paired_significance_test(
            df_a[metric].tolist(), df_b[metric].tolist(),
            f"{label_a} ({metric})", f"{label_b} ({metric})",
        )
        wins_a = int((df_a[metric].values > df_b[metric].values).sum())
        n = len(df_a)
        summary_rows.append({
            "comparison": comparison_name,
            "metric": metric,
            f"mean_{label_a}": df_a[metric].mean(),
            f"mean_{label_b}": df_b[metric].mean(),
            "mean_diff": (df_a[metric] - df_b[metric]).mean(),
            "t_stat": t_stat,
            "p_value": p_value,
            "significant_at_0.05": bool(p_value < 0.05),
            f"{label_a}_win_ratio": f"{wins_a}/{n}",
        })


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = load_elliptic(
        features_path=cfg["features_path"], edges_path=cfg["edges_path"], classes_path=cfg["classes_path"],
        train_end=cfg["train_end"], val_end=cfg["val_end"], device=device,
    )
    x, y = data["x"], data["y"]
    train_mask, val_mask, test_mask = data["train_mask"], data["val_mask"], data["test_mask"]

    seeds = cfg["seeds"]

    rows = []
    for seed in seeds:
        metrics = train_eval_xgb_raw(seed, x, y, train_mask, val_mask, test_mask)
        metrics["seed"] = seed
        rows.append(metrics)
        print(f"seed={seed:5d}  F1={metrics['F1']:.4f}  AUC={metrics['AUC']:.4f}  threshold={metrics['threshold']:.2f}")

        safe_log_run(phase="elliptic", experiment="11_ablation_xgboost_raw_features_15seed", seed=seed,
                     metric_name="F1", metric_value=metrics["F1"],
                     checkpoint_path="", log_path=f"{LOGS_DIR}/elliptic_11.log")

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(f"{METRICS_DIR}/elliptic_xgboost_raw_only_15seed_results.csv", index=False)
    print(f"\nذخیره شد: {METRICS_DIR}/elliptic_xgboost_raw_only_15seed_results.csv")

    df_sage = load_reference(GRAPHSAGE_REFERENCE_CSV, seeds, "GraphSAGE")
    df_hybrid = load_reference(HYBRID_REFERENCE_CSV, seeds, "Hybrid")

    summary_rows = []

    compare(df_raw, df_sage, "xgb_raw", "graphsage",
            summary_rows, "XGBoost (raw only) در برابر GraphSAGE")
    compare(df_raw, df_hybrid, "xgb_raw", "hybrid",
            summary_rows, "XGBoost (raw only) در برابر Hybrid (GraphSAGE+XGBoost)")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(f"{METRICS_DIR}/elliptic_xgboost_ablation_15seed_summary.csv", index=False)

    print(f"\n{'=' * 70}\nجدول نهایی\n{'=' * 70}")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(df_summary.round(4).to_string(index=False))

    return df_summary


if __name__ == "__main__":
    main()
