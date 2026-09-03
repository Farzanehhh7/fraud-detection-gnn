import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

from fraud_gnn.utils.seeding import set_seed


def evaluate_binary(name, y_true, y_pred, y_prob, verbose=True):
    results = {
        "model": name,
        "AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    if verbose:
        print(f"\n--- {name} ---")
        for k, v in results.items():
            if k != "model":
                print(f"{k:12s}: {v:.4f}")
    return results


def evaluate_gnn(name, model, x, edge_index, y, mask, two_class_softmax=False):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        if two_class_softmax:
            probs = torch.softmax(out, dim=1)[mask, 1].cpu().numpy()
            preds = out[mask].argmax(dim=1).cpu().numpy()
        else:
            probs = torch.sigmoid(out[mask]).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
        y_true = y[mask].cpu().numpy()
    return evaluate_binary(name, y_true, preds, probs)


def get_temporal_split_masks(time_step, y, train_end=27, val_end=34, device=None):
    if not torch.is_tensor(time_step):
        time_step = torch.tensor(time_step)
    if device is not None:
        time_step = time_step.to(device)
        y = y.to(device)

    labeled = y != -1
    train_mask = labeled & (time_step <= train_end)
    val_mask = labeled & (time_step > train_end) & (time_step <= val_end)
    test_mask = labeled & (time_step > val_end)
    return train_mask, val_mask, test_mask


def find_best_threshold(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def run_multi_seed(run_one_seed_fn, seeds=(42, 1, 7, 123, 2024), name="model", verbose=True):
    rows = []
    for seed in seeds:
        set_seed(seed)
        metrics = run_one_seed_fn(seed)
        metrics["seed"] = seed
        rows.append(metrics)

    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if c not in ("model", "seed")]
    summary = df[numeric_cols].agg(["mean", "std"])

    if verbose:
        print(f"\n=== {name} — نتیجه روی {len(seeds)} seed ===")
        print(df[["seed"] + numeric_cols].round(4).to_string(index=False))
        print("\nمیانگین ± انحراف معیار:")
        for c in numeric_cols:
            print(f"{c:12s}: {summary.loc['mean', c]:.4f} ± {summary.loc['std', c]:.4f}")

    return df, summary


class EarlyStopper:
    def __init__(self, patience=20, min_delta=1e-4, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.best_state = None
        self.best_epoch = None
        self.counter = 0
        self.should_stop = False

    def step(self, score, model, epoch=None):
        if self.best_score is None:
            improved = True
        elif self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model


def save_checkpoint(model, path, extra=None):
    payload = {"model_state": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"checkpoint ذخیره شد در {path}")


def load_checkpoint(model, path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state"])
    return payload


def report_at_percentile_thresholds(y_true, y_prob, percentiles=(90, 99, 99.9), verbose=True):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    rows = []
    for pct in percentiles:
        cutoff = np.percentile(y_prob, pct)
        preds = (y_prob >= cutoff).astype(int)
        row = {
            "percentile": pct,
            "threshold": cutoff,
            "n_flagged": int(preds.sum()),
            "precision": precision_score(y_true, preds, zero_division=0),
            "recall": recall_score(y_true, preds, zero_division=0),
        }
        rows.append(row)
        if verbose:
            print(f"صدک {pct:5.1f} | threshold={cutoff:.4f} | flagged={row['n_flagged']:6d} | "
                  f"Precision={row['precision']:.4f} | Recall={row['recall']:.4f}")
    return pd.DataFrame(rows)


def compute_mad_neighbors(embeddings, edge_index):
    src, dst = edge_index
    h_src = embeddings[src]
    h_dst = embeddings[dst]
    cos_sim = torch.nn.functional.cosine_similarity(h_src, h_dst, dim=1)
    distance = 1 - cos_sim
    return distance.mean().item()


def bootstrap_test_ci(y_true, y_prob, n_iterations=100, sample_frac=0.5, threshold=0.5, seed=42, verbose=True):
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    n = len(y_true)
    scores = []
    for _ in range(n_iterations):
        idx = rng.choice(n, size=int(n * sample_frac), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        preds = (y_prob[idx] >= threshold).astype(int)
        scores.append(f1_score(y_true[idx], preds, zero_division=0))
    scores = np.array(scores)
    lower, upper = np.percentile(scores, [2.5, 97.5])
    if verbose:
        print(f"Bootstrap {len(scores)} تکرار معتبر: میانگین F1={scores.mean():.4f}   "
              f"فاصله اطمینان ۹۵٪ = [{lower:.4f}, {upper:.4f}]")
    return scores.mean(), lower, upper


def log_experiment(log_path, run_info):
    import os
    df_row = pd.DataFrame([run_info])
    if os.path.exists(log_path):
        df_row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(log_path, mode="w", header=True, index=False)
    print(f"لاگ شد در {log_path}: {run_info}")


def build_edge_index(node_ids, edge_src_ids, edge_dst_ids):
    map_id = {node_id: i for i, node_id in enumerate(node_ids)}
    src_idx = [map_id[s] for s in edge_src_ids]
    dst_idx = [map_id[d] for d in edge_dst_ids]
    return map_id, torch.tensor([src_idx, dst_idx], dtype=torch.long)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = self.alpha[targets]
        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()
