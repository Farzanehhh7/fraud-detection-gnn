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
RUN_LOG_PATH = os.path.join(RUN_LOGS_DIR, f"elliptic_09_{_timestamp}.log")
_log_file = open(RUN_LOG_PATH, "w", encoding="utf-8")

sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)

print(f"[این اجرا هم‌زمان توی این فایل ذخیره می‌شه: {RUN_LOG_PATH}]")

import atexit
atexit.register(_log_file.close)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch_geometric.nn import GCNConv
from xgboost import XGBClassifier

from fraud_gnn.data.elliptic import load_elliptic
from fraud_gnn.models.archived.attention_variants import GATFraudDetector, ATGATModulated, ATGATGraphSAGE
from fraud_gnn.models.archived.hybrid_and_skipgcn import SkipGCN, EmbeddingExtractor
from fraud_gnn.evaluation.significance_tests import paired_significance_test
from fraud_gnn.utils.metrics import evaluate_binary, find_best_threshold, run_multi_seed
from fraud_gnn.utils.run_logger import log_run

CONFIG_PATH = "configs/elliptic_binary.yaml"
EPOCHS = 200
TEMPORAL_DIM = 32
GRAPHSAGE_REFERENCE_CSV = "outputs/metrics/step32_graphsage_15seed_results.csv"
METRICS = ("AUC", "PR-AUC", "F1", "Precision", "Recall", "MCC")

METRICS_DIR = "outputs/metrics"
LOGS_DIR = "outputs/logs"

RUN_ONLY = ["TripleAttention-structural+global", "Hybrid"]

SKIP_IF_RESULT_EXISTS = True


class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def _tune_and_eval(model_name, probs_all, y, val_mask, test_mask):
    y_val = y[val_mask].cpu().numpy()
    probs_val = probs_all[val_mask.cpu().numpy()]
    best_t, _ = find_best_threshold(y_val, probs_val)

    y_test = y[test_mask].cpu().numpy()
    probs_test = probs_all[test_mask.cpu().numpy()]
    preds_test = (probs_test >= best_t).astype(int)

    metrics = evaluate_binary(model_name, y_test, preds_test, probs_test, verbose=False)
    metrics["threshold"] = best_t
    return metrics


def train_eval_gcn(seed, x, edge_index, y, train_mask, val_mask, test_mask, class_weights, hidden_channels):
    torch.manual_seed(seed)
    model = SimpleGCN(in_channels=x.shape[1], hidden_channels=hidden_channels, out_channels=2).to(x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
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

    return _tune_and_eval("GCN", probs_all, y, val_mask, test_mask)


def train_eval_gat(seed, x, edge_index, y, train_mask, val_mask, test_mask, class_weights, hidden_channels):
    torch.manual_seed(seed)
    model = GATFraudDetector(in_channels=x.shape[1], hidden_channels=hidden_channels // 2, out_channels=2).to(x.device)
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

    return _tune_and_eval("GAT", probs_all, y, val_mask, test_mask)


def train_eval_skipgcn(seed, x, edge_index, y, train_mask, val_mask, test_mask, class_weights, hidden_channels):
    torch.manual_seed(seed)
    model = SkipGCN(in_channels=x.shape[1], hidden_channels=hidden_channels, out_channels=2).to(x.device)
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

    return _tune_and_eval("Skip-GCN", probs_all, y, val_mask, test_mask)


def train_eval_film(seed, x, edge_index, time_steps, num_timesteps, y, train_mask, val_mask, test_mask,
                     class_weights, hidden_channels):
    torch.manual_seed(seed)
    model = ATGATModulated(in_channels=x.shape[1], hidden_channels=hidden_channels, out_channels=2,
                            num_timesteps=num_timesteps, temporal_dim=TEMPORAL_DIM).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index, time_steps)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, time_steps)
        probs_all = F.softmax(out, dim=1)[:, 1].cpu().numpy()

    return _tune_and_eval("FiLM modulated", probs_all, y, val_mask, test_mask)


def train_eval_triple_attention(seed, x, edge_index, time_steps, num_timesteps, y, train_mask, val_mask, test_mask,
                                 class_weights, hidden_channels, use_temporal, use_global):
    torch.manual_seed(seed)
    model = ATGATGraphSAGE(
        in_channels=x.shape[1], hidden_channels=hidden_channels, out_channels=2,
        num_timesteps=num_timesteps, temporal_dim=TEMPORAL_DIM,
        use_temporal=use_temporal, use_global=use_global,
    ).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out, _ = model(x, edge_index, time_steps)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out, _ = model(x, edge_index, time_steps)
        probs_all = F.softmax(out, dim=1)[:, 1].cpu().numpy()

    return _tune_and_eval("Triple Attention", probs_all, y, val_mask, test_mask)


def train_eval_hybrid(seed, x, edge_index, y, train_mask, val_mask, test_mask, hidden_channels, embedding_dim=64):
    torch.manual_seed(seed)
    emb_model = EmbeddingExtractor(in_channels=x.shape[1], hidden_channels=hidden_channels,
                                    embedding_dim=embedding_dim).to(x.device)
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

    metrics = evaluate_binary("Hybrid (GraphSAGE+XGBoost)", y_test, preds_test, probs_test, verbose=False)
    metrics["threshold"] = best_t
    return metrics


def load_graphsage_reference(seeds):
    df = pd.read_csv(GRAPHSAGE_REFERENCE_CSV)
    df = df.set_index("seed").loc[list(seeds)].reset_index()
    return df


def compare_to_graphsage(df_variant, df_sage, variant_label, summary_rows):

    df_sage = df_sage.sort_values("seed").reset_index(drop=True)
    df_variant = df_variant.sort_values("seed").reset_index(drop=True)

    if not (df_sage["seed"].values == df_variant["seed"].values).all():
        raise ValueError(
            f"seed mismatch بین GraphSAGE و {variant_label} — مقایسه‌ی element-wise معتبر نیست."
        )

    print(f"\n{'=' * 70}\nمقایسه‌ی آماری: {variant_label} در برابر GraphSAGE (۱۵ seed)\n{'=' * 70}")
    for metric in METRICS:
        t_stat, p_value = paired_significance_test(
            df_sage[metric].tolist(), df_variant[metric].tolist(),
            f"GraphSAGE ({metric})", f"{variant_label} ({metric})",
        )
        wins_sage = int((df_sage[metric].values > df_variant[metric].values).sum())
        n = len(df_sage)
        summary_rows.append({
            "architecture": variant_label,
            "metric": metric,
            "mean_graphsage": df_sage[metric].mean(),
            "mean_variant": df_variant[metric].mean(),
            "mean_diff": (df_sage[metric] - df_variant[metric]).mean(),
            "t_stat": t_stat,
            "p_value": p_value,
            "significant_at_0.05": bool(p_value < 0.05),
            "graphsage_win_ratio": f"{wins_sage}/{n}",
        })


def safe_log_run(**kwargs):

    try:
        log_run(**kwargs)
    except Exception as exc:
        print(f"[هشدار] log_run شکست خورد ({kwargs.get('experiment')}, seed={kwargs.get('seed')}): {exc}")


def csv_path_for(label):
    return f"{METRICS_DIR}/elliptic_{label.lower().replace('+', '_')}_15seed_results.csv"


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
    x, y, edge_index = data["x"], data["y"], data["edge_index"]
    train_mask, val_mask, test_mask = data["train_mask"], data["val_mask"], data["test_mask"]
    class_weights = data["class_weights"]
    time_steps = (data["time_steps_raw"] - 1).to(device)
    num_timesteps = int(time_steps.max().item()) + 1

    seeds = cfg["seeds"]
    hidden = cfg["hidden_channels"]

    df_sage = load_graphsage_reference(seeds)
    summary_rows = []

    variants = {
        "GCN": lambda: run_multi_seed(
            lambda seed: train_eval_gcn(seed, x, edge_index, y, train_mask, val_mask, test_mask,
                                         class_weights, hidden),
            seeds=seeds, name="GCN"),
        "GAT": lambda: run_multi_seed(
            lambda seed: train_eval_gat(seed, x, edge_index, y, train_mask, val_mask, test_mask,
                                         class_weights, hidden),
            seeds=seeds, name="GAT"),
        "Skip-GCN": lambda: run_multi_seed(
            lambda seed: train_eval_skipgcn(seed, x, edge_index, y, train_mask, val_mask, test_mask,
                                             class_weights, hidden),
            seeds=seeds, name="Skip-GCN"),
        "FiLM+global": lambda: run_multi_seed(
            lambda seed: train_eval_film(seed, x, edge_index, time_steps, num_timesteps, y,
                                          train_mask, val_mask, test_mask, class_weights, hidden),
            seeds=seeds, name="FiLM+global"),
        "TripleAttention-raw": lambda: run_multi_seed(
            lambda seed: train_eval_triple_attention(seed, x, edge_index, time_steps, num_timesteps, y,
                                                       train_mask, val_mask, test_mask, class_weights, hidden,
                                                       use_temporal=True, use_global=True),
            seeds=seeds, name="Triple Attention (raw, سه‌جریانی)"),
        "TripleAttention-structural+temporal": lambda: run_multi_seed(
            lambda seed: train_eval_triple_attention(seed, x, edge_index, time_steps, num_timesteps, y,
                                                       train_mask, val_mask, test_mask, class_weights, hidden,
                                                       use_temporal=True, use_global=False),
            seeds=seeds, name="structural+temporal فقط"),
        "TripleAttention-structural+global": lambda: run_multi_seed(
            lambda seed: train_eval_triple_attention(seed, x, edge_index, time_steps, num_timesteps, y,
                                                       train_mask, val_mask, test_mask, class_weights, hidden,
                                                       use_temporal=False, use_global=True),
            seeds=seeds, name="structural+global فقط (اصلاح‌شده)"),
        "Hybrid": lambda: run_multi_seed(
            lambda seed: train_eval_hybrid(seed, x, edge_index, y, train_mask, val_mask, test_mask, hidden),
            seeds=seeds, name="Hybrid (GraphSAGE+XGBoost)"),
    }

    if RUN_ONLY is not None:
        variants = {k: v for k, v in variants.items() if k in RUN_ONLY}

    for label, run_fn in variants.items():
        csv_name = csv_path_for(label)

        if SKIP_IF_RESULT_EXISTS and os.path.exists(csv_name):
            print(f"\n[رد شد] نتیجه‌ی '{label}' از قبل توی {csv_name} موجوده — دوباره ران نمی‌شه.")
            df_variant = pd.read_csv(csv_name)
        else:
            df_variant, _ = run_fn()
            df_variant.to_csv(csv_name, index=False)

            for seed, row in zip(seeds, df_variant.to_dict("records")):
                safe_log_run(phase="elliptic", experiment=f"09_significance_test_architectures_15seed_{label}",
                             seed=seed, metric_name="F1", metric_value=row["F1"],
                             checkpoint_path="", log_path=f"{LOGS_DIR}/elliptic_09.log")

        compare_to_graphsage(df_variant, df_sage, label, summary_rows)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(f"{METRICS_DIR}/elliptic_architecture_comparison_15seed_summary.csv", index=False)

    print(f"\n{'=' * 70}\nجدول نهایی مقایسه‌ی همه‌ی معماری‌ها در برابر GraphSAGE\n{'=' * 70}")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(df_summary.round(4).to_string(index=False))

    return df_summary


if __name__ == "__main__":
    main()