import time

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

from fraud_gnn.utils.metrics import build_edge_index

SENDER_COL, RECEIVER_COL = "Sender_account", "Receiver_account"
LABEL_COL, TYPE_COL = "Is_laundering", "Laundering_type"
AMOUNT_COL, PTYPE_COL = "Amount", "Payment_type"

FAMILY_MAP = {
    0: [2, 4, 6, 7, 8, 9, 10, 12, 15],
    1: [3, 11, 13, 14, 16],
    2: [0, 1, 5],
}
FAMILY_NAMES = {
    0: "Structural/Multi-hop",
    1: "Volume/Threshold",
    2: "Behavioral/Temporal",
}


def build_family_labels(y_type, family_map=None):
    family_map = family_map or FAMILY_MAP
    y_family = torch.full_like(y_type, -1)
    for fam_id, orig_ids in family_map.items():
        for orig_id in orig_ids:
            y_family[y_type == orig_id] = fam_id
    return y_family


def build_samld_graph(file_path="datasets/SAML-D.csv", n_rows=1_000_000,
                       output_prefix="samld_processed_v3", save=True):
    print(f"در حال بارگذاری {n_rows} ردیف اول از SAML-D...")
    df = pd.read_csv(file_path, nrows=n_rows)
    print("ستون‌های موجود:", list(df.columns))

    date_col = None
    for candidate in ["Date", "Time", "Date_time", "Timestamp"]:
        if candidate in df.columns:
            date_col = candidate
            break

    has_temporal_cutoff = False
    if date_col is not None:
        df["_dt"] = pd.to_datetime(df[date_col], errors="coerce")
        if df["_dt"].isna().mean() < 0.05:
            df = df.sort_values("_dt").reset_index(drop=True)
            n = len(df)
            cutoff70_dt = df.loc[int(n * 0.70), "_dt"]
            cutoff85_dt = df.loc[int(n * 0.85), "_dt"]
            df["pre_cutoff"] = df["_dt"] <= cutoff70_dt
            has_temporal_cutoff = True
        else:
            date_col = None

    if not has_temporal_cutoff:
        raise RuntimeError(
            "بدون ستون تاریخ معتبر، تقسیم بر اساس آخرین فعالیت اصلاً معنا ندارد."
        )

    all_accounts = pd.unique(df[[SENDER_COL, RECEIVER_COL]].values.ravel())
    acc_map, edge_index = build_edge_index(all_accounts, df[SENDER_COL], df[RECEIVER_COL])

    last_seen = pd.concat([
        df[[SENDER_COL, "_dt"]].rename(columns={SENDER_COL: "account"}),
        df[[RECEIVER_COL, "_dt"]].rename(columns={RECEIVER_COL: "account"}),
    ]).groupby("account")["_dt"].max()
    last_seen = last_seen.reindex(all_accounts)

    split_series = pd.Series("test", index=all_accounts)
    split_series[last_seen <= cutoff70_dt] = "train"
    split_series[(last_seen > cutoff70_dt) & (last_seen <= cutoff85_dt)] = "val"

    train_mask = torch.tensor((split_series == "train").values)
    val_mask = torch.tensor((split_series == "val").values)
    test_mask = torch.tensor((split_series == "test").values)

    df_hist = df[df["pre_cutoff"]].copy()

    sent_stats = df_hist.groupby(SENDER_COL)[AMOUNT_COL].agg(["sum", "mean", "count"])
    sent_stats.columns = ["sent_amount_sum", "sent_amount_mean", "sent_amount_count"]
    sent_ptype = df_hist.groupby(SENDER_COL)[PTYPE_COL].nunique().rename("sent_payment_type_nunique")

    recv_stats = df_hist.groupby(RECEIVER_COL)[AMOUNT_COL].agg(["sum", "mean", "count"])
    recv_stats.columns = ["recv_amount_sum", "recv_amount_mean", "recv_amount_count"]
    recv_ptype = df_hist.groupby(RECEIVER_COL)[PTYPE_COL].nunique().rename("recv_payment_type_nunique")

    feat_df = pd.concat([sent_stats, sent_ptype, recv_stats, recv_ptype], axis=1)
    feat_df = feat_df.reindex(all_accounts).fillna(0.0)
    feature_cols = list(feat_df.columns)

    cold_start = (feat_df[["sent_amount_count", "recv_amount_count"]].sum(axis=1) == 0)

    scaler = StandardScaler()
    scaler.fit(feat_df.values[train_mask.numpy()])
    x = scaler.transform(feat_df.values)
    x = torch.tensor(x, dtype=torch.float)

    illicit_tx = df[df[LABEL_COL] == 1]
    illicit_accounts = set(illicit_tx[SENDER_COL]).union(set(illicit_tx[RECEIVER_COL]))

    binary_label = pd.Series(0, index=all_accounts)
    binary_label.loc[list(illicit_accounts & set(all_accounts))] = 1

    type_by_account = {}
    for acc in illicit_accounts:
        mask = (illicit_tx[SENDER_COL] == acc) | (illicit_tx[RECEIVER_COL] == acc)
        types = illicit_tx.loc[mask, TYPE_COL]
        if len(types) > 0:
            type_by_account[acc] = types.mode().iloc[0]

    le_type = LabelEncoder()
    all_types = sorted(set(type_by_account.values()))
    le_type.fit(all_types)

    type_label = pd.Series(-1, index=all_accounts)
    for acc, t in type_by_account.items():
        type_label.loc[acc] = le_type.transform([t])[0]

    y_binary = torch.tensor(binary_label.loc[all_accounts].values, dtype=torch.long)
    y_type = torch.tensor(type_label.loc[all_accounts].values, dtype=torch.long)

    result = {
        "x": x,
        "edge_index": edge_index,
        "y_binary": y_binary,
        "y_type": y_type,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "feature_cols": feature_cols,
        "num_types": len(all_types),
        "account_ids": list(all_accounts),
        "cold_start_mask": torch.tensor(cold_start.reindex(all_accounts).values),
        "has_temporal_cutoff": has_temporal_cutoff,
        "scaler": scaler,
        "label_encoder": le_type,
        "account_map": acc_map,
    }

    if save:
        torch.save({k: v for k, v in result.items() if k not in ("scaler", "label_encoder", "account_map")},
                    f"{output_prefix}.pt")
        joblib.dump(scaler, "samld_scaler_v3.pkl")
        print(f"همه چیز ذخیره شد در {output_prefix}.pt")

    return result


def compute_structural_features(edge_index, num_nodes):
    t0 = time.time()
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()

    in_degree = np.bincount(dst, minlength=num_nodes).astype(np.float32)
    out_degree = np.bincount(src, minlength=num_nodes).astype(np.float32)

    edge_set = set(zip(src.tolist(), dst.tolist()))
    reciprocal_mask = np.fromiter(
        ((d, s) in edge_set for s, d in zip(src.tolist(), dst.tolist())),
        dtype=bool, count=len(src),
    )
    recip_src = src[reciprocal_mask]
    recip_dst = dst[reciprocal_mask]
    reciprocal_count = (np.bincount(recip_src, minlength=num_nodes) +
                         np.bincount(recip_dst, minlength=num_nodes)).astype(np.float32)

    gather_scatter_score = np.minimum(in_degree, out_degree)
    degree_ratio = out_degree / (in_degree + 1.0)
    is_pass_through = ((in_degree >= 2) & (out_degree >= 2)).astype(np.float32)

    features = np.stack(
        [in_degree, out_degree, gather_scatter_score, degree_ratio, reciprocal_count, is_pass_through],
        axis=1,
    )
    print(f"Structural features computed in {time.time() - t0:.2f}s   shape={features.shape}")
    return features, [
        "in_degree", "out_degree", "gather_scatter_score",
        "degree_ratio", "reciprocal_count", "is_pass_through",
    ]
