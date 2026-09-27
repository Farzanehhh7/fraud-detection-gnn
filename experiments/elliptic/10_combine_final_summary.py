import os
import sys
import io
import glob

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import pandas as pd
from fraud_gnn.evaluation.significance_tests import paired_significance_test

METRICS_DIR = "outputs/metrics"
GRAPHSAGE_REFERENCE_CSV = f"{METRICS_DIR}/step32_graphsage_15seed_results.csv"
METRICS = ("AUC", "PR-AUC", "F1", "Precision", "Recall", "MCC")

ARCHITECTURE_FILES = {
    "GCN":                                  "elliptic_gcn_15seed_results.csv",
    "GAT":                                  "elliptic_gat_15seed_results.csv",
    "Skip-GCN":                             "elliptic_skip-gcn_15seed_results.csv",
    "FiLM+global":                          "elliptic_film_global_15seed_results.csv",
    "TripleAttention-raw":                  "elliptic_tripleattention-raw_15seed_results.csv",
    "TripleAttention-structural+temporal":  "elliptic_tripleattention-structural_temporal_15seed_results.csv",
    "TripleAttention-structural+global":    "elliptic_tripleattention-structural_global_15seed_results.csv",
    "Hybrid":                               "elliptic_hybrid_15seed_results.csv",
}


def load_and_align(path, seeds):
    df = pd.read_csv(path)
    df = df[df["seed"].isin(seeds)].sort_values("seed").reset_index(drop=True)
    return df


def main():
    if not os.path.exists(GRAPHSAGE_REFERENCE_CSV):
        raise FileNotFoundError(f"فایل مرجع GraphSAGE پیدا نشد: {GRAPHSAGE_REFERENCE_CSV}")

    df_sage_full = pd.read_csv(GRAPHSAGE_REFERENCE_CSV)
    seeds = sorted(df_sage_full["seed"].tolist())
    df_sage = load_and_align(GRAPHSAGE_REFERENCE_CSV, seeds)

    summary_rows = []
    missing = []

    for label, filename in ARCHITECTURE_FILES.items():
        path = os.path.join(METRICS_DIR, filename)
        if not os.path.exists(path):
            missing.append((label, path))
            continue

        df_variant = load_and_align(path, seeds)

        if not (df_sage["seed"].values == df_variant["seed"].values).all():
            print(f"[هشدار] عدم تطابق seed برای {label} — این معماری رد شد.")
            continue

        print(f"\n{'=' * 70}\n{label} در برابر GraphSAGE\n{'=' * 70}")
        for metric in METRICS:
            t_stat, p_value = paired_significance_test(
                df_sage[metric].tolist(), df_variant[metric].tolist(),
                f"GraphSAGE ({metric})", f"{label} ({metric})",
            )
            wins_sage = int((df_sage[metric].values > df_variant[metric].values).sum())
            n = len(df_sage)
            summary_rows.append({
                "architecture": label,
                "metric": metric,
                "mean_graphsage": df_sage[metric].mean(),
                "mean_variant": df_variant[metric].mean(),
                "mean_diff": (df_sage[metric] - df_variant[metric]).mean(),
                "t_stat": t_stat,
                "p_value": p_value,
                "significant_at_0.05": bool(p_value < 0.05),
                "graphsage_win_ratio": f"{wins_sage}/{n}",
            })

    if missing:
        print(f"\n{'=' * 70}\nمعماری‌هایی که CSV نتیجه‌شون پیدا نشد و رد شدن:\n{'=' * 70}")
        for label, path in missing:
            print(f"  - {label}  (انتظار می‌رفت اینجا باشه: {path})")

    df_summary = pd.DataFrame(summary_rows)
    out_path = f"{METRICS_DIR}/elliptic_architecture_comparison_FINAL_ALL.csv"
    df_summary.to_csv(out_path, index=False)

    print(f"\n{'=' * 70}\nجدول نهایی مقایسه‌ی همه‌ی معماری‌های موجود در برابر GraphSAGE\n{'=' * 70}")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(df_summary.round(4).to_string(index=False))

    print(f"\n[ذخیره شد در: {out_path}]")
    return df_summary


if __name__ == "__main__":
    main()
