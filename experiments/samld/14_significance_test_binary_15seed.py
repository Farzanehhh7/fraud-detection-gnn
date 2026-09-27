

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
RUN_LOG_PATH = os.path.join(RUN_LOGS_DIR, f"samld_14_{_timestamp}.log")
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

import pandas as pd
import torch
import yaml

from fraud_gnn.models.graphsage_samld import StructuralOnlyGraphSAGE
from fraud_gnn.evaluation.significance_tests import paired_significance_test
from fraud_gnn.utils.metrics import evaluate_binary, find_best_threshold, load_checkpoint
from fraud_gnn.utils.run_logger import log_run

CONFIG_PATH = "configs/samld_binary.yaml"
METRICS = ("AUC", "PR-AUC", "F1", "Precision", "Recall", "MCC")
METRICS_DIR = "outputs/metrics"
LOGS_DIR = "outputs/logs"
LR_CSV = f"{METRICS_DIR}/samld_lr_nograph_15seed_results.csv"
RF_CSV = f"{METRICS_DIR}/samld_rf_nograph_15seed_results.csv"


def safe_log_run(**kwargs):

    try:
        log_run(**kwargs)
    except Exception as exc:
        print(f"[هشدار] log_run شکست خورد (seed={kwargs.get('seed')}): {exc}")


def recompute_graphsage_results(cfg, device):

    print("Loading real SAML-D v3 data...")
    data_dict = torch.load(f"{cfg['output_prefix']}.pt", map_location="cpu", weights_only=False)
    x = data_dict["x"]
    edge_index = data_dict["edge_index"]
    y = data_dict["y_binary"]
    val_mask = data_dict["val_mask"]
    test_mask = data_dict["test_mask"]

    rows = []
    for seed in cfg["seeds"]:
        model = StructuralOnlyGraphSAGE(
            in_channels=x.shape[1], hidden_channels=cfg["hidden_channels"], out_channels=cfg["out_channels"],
        ).to(device)
        ckpt_path = f"{cfg['checkpoint_dir']}/samld_seed_{seed}.pt"
        load_checkpoint(model, ckpt_path, map_location=device)

        model.eval()
        with torch.no_grad():
            out = model(x.to(device), edge_index.to(device))
            probs_all = torch.softmax(out, dim=1)[:, 1].cpu().numpy()

        y_val = y[val_mask].numpy()
        probs_val = probs_all[val_mask.numpy()]
        best_t, _ = find_best_threshold(y_val, probs_val)

        y_test = y[test_mask].numpy()
        probs_test = probs_all[test_mask.numpy()]
        preds_test = (probs_test >= best_t).astype(int)

        metrics = evaluate_binary("SAML-D GraphSAGE", y_test, preds_test, probs_test, verbose=False)
        metrics["threshold"] = best_t
        metrics["seed"] = seed
        rows.append(metrics)

        print(f"seed={seed:5d}  F1={metrics['F1']:.4f}  AUC={metrics['AUC']:.4f}  "
              f"threshold={best_t:.2f}")

        safe_log_run(phase="samld", experiment="14_significance_test_binary_15seed", seed=seed,
                     metric_name="F1", metric_value=metrics["F1"],
                     checkpoint_path=ckpt_path, log_path=f"{LOGS_DIR}/samld_14.log")

    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if c not in ("model", "seed")]
    print("\nمیانگین ± انحراف معیار (بازتولیدشده از checkpoint ها):")
    for c in numeric_cols:
        print(f"{c:12s}: {df[c].mean():.4f} ± {df[c].std():.4f}")

    return df


def compare(df_sage, df_baseline, baseline_label, seeds, summary_rows):
    df_baseline = df_baseline.set_index("seed").loc[list(seeds)].reset_index()

    print(f"\n{'=' * 70}\nGraphSAGE در برابر {baseline_label} (۱۵ seed، SAML-D دوتایی)\n{'=' * 70}")
    for metric in METRICS:
        t_stat, p_value = paired_significance_test(
            df_sage[metric].tolist(), df_baseline[metric].tolist(),
            f"GraphSAGE ({metric})", f"{baseline_label} ({metric})",
        )
        wins_sage = int((df_sage[metric].values > df_baseline[metric].values).sum())
        n = len(df_sage)
        summary_rows.append({
            "baseline": baseline_label,
            "metric": metric,
            "mean_graphsage": df_sage[metric].mean(),
            "mean_baseline": df_baseline[metric].mean(),
            "mean_diff": (df_sage[metric] - df_baseline[metric]).mean(),
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

    df_sage = recompute_graphsage_results(cfg, device)
    df_sage.to_csv(f"{METRICS_DIR}/samld_graphsage_binary_15seed_results.csv", index=False)
    print(f"\nذخیره شد: {METRICS_DIR}/samld_graphsage_binary_15seed_results.csv")

    if not os.path.exists(LR_CSV):
        raise FileNotFoundError(f"فایل baseline پیدا نشد: {LR_CSV} — اول 13_binary_baseline_multiseed.py رو اجرا کن.")
    if not os.path.exists(RF_CSV):
        raise FileNotFoundError(f"فایل baseline پیدا نشد: {RF_CSV} — اول 13_binary_baseline_multiseed.py رو اجرا کن.")

    df_lr = pd.read_csv(LR_CSV)
    df_rf = pd.read_csv(RF_CSV)

    summary_rows = []
    compare(df_sage, df_lr, "LR بدون گراف", seeds, summary_rows)
    compare(df_sage, df_rf, "RF بدون گراف", seeds, summary_rows)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(f"{METRICS_DIR}/samld_binary_graphsage_vs_nograph_15seed_summary.csv", index=False)

    print(f"\n{'=' * 70}\nجدول نهایی\n{'=' * 70}")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(df_summary.round(4).to_string(index=False))

    return df_summary


if __name__ == "__main__":
    main()