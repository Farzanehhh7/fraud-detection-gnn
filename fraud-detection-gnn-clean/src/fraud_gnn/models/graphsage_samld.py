import json
import os
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

TYPE_NAMES_PATH = "samld_type_names.json"
MIN_TRAIN_SAMPLES_PER_TYPE = 10


class StructuralOnlyGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, return_embeddings=False):
        h1 = F.dropout(F.relu(self.conv1(x, edge_index)), p=0.3, training=self.training)
        h2 = F.dropout(F.relu(self.conv2(h1, edge_index)), p=0.3, training=self.training)
        out = self.classifier(h2)
        if return_embeddings:
            return out, h1, h2
        return out


def load_type_names(path=TYPE_NAMES_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def merge_rare_types(y_type, train_mask, min_train_samples=MIN_TRAIN_SAMPLES_PER_TYPE):
    train_types = y_type[train_mask]
    train_types = train_types[train_types != -1].tolist()
    counts = Counter(train_types)

    all_type_ids = sorted(set(y_type[y_type != -1].tolist()))
    kept_type_ids = [t for t in all_type_ids if counts.get(t, 0) >= min_train_samples]
    rare_type_ids = [t for t in all_type_ids if t not in kept_type_ids]

    remap = {t: i for i, t in enumerate(kept_type_ids)}
    other_id = len(kept_type_ids)
    if rare_type_ids:
        for t in rare_type_ids:
            remap[t] = other_id
        num_new_types = other_id + 1
    else:
        num_new_types = other_id

    print(f"Types kept as-is: {len(kept_type_ids)}   Types merged into 'Other': {len(rare_type_ids)}")
    return remap, num_new_types, kept_type_ids


def extract_embeddings(model, x, edge_index, device, batch_eval_size=50000):
    model.eval()
    with torch.no_grad():
        _, _, h2_full = model(x.to(device), edge_index.to(device), return_embeddings=True)
    return h2_full.cpu()


def fit_and_evaluate_type_classifier(h2, y_type_remapped, train_mask, test_mask, num_new_types,
                                      seed_label=None, verbose=True, type_names=None):
    illicit_train = train_mask & (y_type_remapped != -1)
    illicit_test = test_mask & (y_type_remapped != -1)

    X_train = h2[illicit_train].numpy()
    y_train = y_type_remapped[illicit_train].numpy()
    X_test = h2[illicit_test].numpy()
    y_test = y_type_remapped[illicit_test].numpy()

    if verbose:
        tag = f" (seed={seed_label})" if seed_label is not None else ""
        print(f"\nType classifier{tag}: train n={len(y_train)}   test n={len(y_test)}   "
              f"num_types(after merge)={num_new_types}")

    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    labels_present = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    target_names = [type_names.get(i, f"type_{i}") for i in labels_present] if type_names else None
    if verbose:
        print(classification_report(
            y_test, y_pred, labels=labels_present, target_names=target_names, zero_division=0,
        ))

    seed_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    return clf, seed_metrics
