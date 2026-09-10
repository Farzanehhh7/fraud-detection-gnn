import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fraud_gnn.utils.metrics import get_temporal_split_masks, build_edge_index


def load_elliptic(features_path="datasets/elliptic_txs_features.csv",
                   edges_path="datasets/elliptic_txs_edgelist.csv",
                   classes_path="datasets/elliptic_txs_classes.csv",
                   train_end=27, val_end=34, device=None):
    df_feat = pd.read_csv(features_path, header=None)
    df_edge = pd.read_csv(edges_path)
    df_class = pd.read_csv(classes_path)

    df_feat.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(165)]
    df_class.columns = ["txId", "class"]
    df_class["label"] = df_class["class"].map({"1": 1, "2": 0, "unknown": -1})

    map_id, edge_index_cpu = build_edge_index(df_feat["txId"].values, df_edge["txId1"], df_edge["txId2"])
    edge_index = edge_index_cpu.to(device) if device is not None else edge_index_cpu

    y_cpu = torch.tensor(df_class["label"].values, dtype=torch.long)
    time_steps_raw = torch.tensor(df_feat["time_step"].values, dtype=torch.long)

    train_mask_cpu, val_mask_cpu, test_mask_cpu = get_temporal_split_masks(
        time_steps_raw, y_cpu, train_end=train_end, val_end=val_end, device=None,
    )

    x_raw = df_feat.drop(columns=["txId", "time_step"]).values
    scaler = StandardScaler()
    scaler.fit(x_raw[train_mask_cpu.numpy()])
    x = torch.tensor(scaler.transform(x_raw), dtype=torch.float)

    y = y_cpu
    train_mask, val_mask, test_mask = train_mask_cpu, val_mask_cpu, test_mask_cpu
    if device is not None:
        x, y = x.to(device), y.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)

    n_pos = (y[train_mask] == 1).sum().item()
    n_neg = (y[train_mask] == 0).sum().item()
    class_weights = torch.tensor([1.0, n_neg / n_pos])
    if device is not None:
        class_weights = class_weights.to(device)

    return {
        "x": x,
        "y": y,
        "edge_index": edge_index,
        "edge_index_cpu": edge_index_cpu,
        "time_steps_raw": time_steps_raw,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "class_weights": class_weights,
        "df_feat": df_feat,
        "df_edge": df_edge,
        "df_class": df_class,
        "map_id": map_id,
    }