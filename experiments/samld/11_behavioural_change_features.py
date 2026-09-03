import joblib
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import yaml

CONFIG_PATH = "configs/samld_binary.yaml"
OUTPUT_PATH = "samld_behavioural_features.pt"
SCALER_OUTPUT_PATH = "samld_behavioural_scaler.pkl"
HIGH_RISK_LOCATIONS = {"Mexico", "Turkey", "Morocco", "UAE"}
SENDER_COL, RECEIVER_COL = "Sender_account", "Receiver_account"
SENDER_LOC, RECEIVER_LOC = "Sender_bank_location", "Receiver_bank_location"
AMOUNT_COL = "Amount"


def location_risk_flags(sub_df):
    sent = sub_df[[SENDER_COL, RECEIVER_LOC]].rename(
        columns={SENDER_COL: "account", RECEIVER_LOC: "counterparty_loc"})
    recv = sub_df[[RECEIVER_COL, SENDER_LOC]].rename(
        columns={RECEIVER_COL: "account", SENDER_LOC: "counterparty_loc"})
    combined = pd.concat([sent, recv], ignore_index=True)
    combined["is_high_risk"] = combined["counterparty_loc"].isin(HIGH_RISK_LOCATIONS)
    return combined.groupby("account")["is_high_risk"].mean()


def counterparties(sub_df):
    sent = sub_df[[SENDER_COL, RECEIVER_COL]].rename(
        columns={SENDER_COL: "account", RECEIVER_COL: "counterparty"})
    recv = sub_df[[RECEIVER_COL, SENDER_COL]].rename(
        columns={RECEIVER_COL: "account", SENDER_COL: "counterparty"})
    combined = pd.concat([sent, recv], ignore_index=True)
    return combined.groupby("account")["counterparty"].apply(set)


def avg_amount(sub_df):
    sent = sub_df.groupby(SENDER_COL)[AMOUNT_COL].mean()
    recv = sub_df.groupby(RECEIVER_COL)[AMOUNT_COL].mean()
    return pd.concat([sent, recv]).groupby(level=0).mean()


def compute_behavioural_features(file_path, n_rows):
    print(f"Loading {n_rows} rows of {file_path}...")
    df = pd.read_csv(file_path, nrows=n_rows)

    date_col = None
    for candidate in ["Date", "Time", "Date_time", "Timestamp"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        raise RuntimeError("No date column found -- cannot compute temporal deviation features.")

    df["_dt"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values("_dt").reset_index(drop=True)
    n = len(df)

    cutoff35_dt = df.loc[int(n * 0.35), "_dt"]
    cutoff70_dt = df.loc[int(n * 0.70), "_dt"]
    print(f"early/late split at 35%: {cutoff35_dt}   feature cutoff at 70%: {cutoff70_dt}")

    df_pre = df[df["_dt"] <= cutoff70_dt].copy()
    df_early = df_pre[df_pre["_dt"] <= cutoff35_dt]
    df_late = df_pre[df_pre["_dt"] > cutoff35_dt]

    all_accounts = pd.unique(df[[SENDER_COL, RECEIVER_COL]].values.ravel())
    print(f"{len(all_accounts)} unique accounts")

    high_risk_ratio = location_risk_flags(df_pre).reindex(all_accounts).fillna(0.0)

    early_cp = counterparties(df_early)
    late_cp = counterparties(df_late)

    novelty = {}
    for acc in all_accounts:
        late_set = late_cp.get(acc, set())
        if not late_set:
            novelty[acc] = 0.0
            continue
        early_set = early_cp.get(acc, set())
        novelty[acc] = len(late_set - early_set) / len(late_set)
    novelty_ratio = pd.Series(novelty).reindex(all_accounts).fillna(0.0)

    early_amt = avg_amount(df_early).reindex(all_accounts)
    late_amt = avg_amount(df_late).reindex(all_accounts)
    deviation = ((late_amt - early_amt).abs() / (early_amt.fillna(0) + 1.0)).fillna(0.0)

    feat_df = pd.DataFrame({
        "high_risk_location_ratio": high_risk_ratio,
        "counterparty_novelty_ratio": novelty_ratio,
        "amount_deviation_ratio": deviation,
    }).reindex(all_accounts).fillna(0.0)

    return feat_df


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    feat_df = compute_behavioural_features(cfg["file_path"], cfg["n_rows"])

    print(f"\nLoading {cfg['output_prefix']}.pt to align account order and get train_mask...")
    data = torch.load(f"{cfg['output_prefix']}.pt", map_location="cpu", weights_only=False)
    saved_account_ids = data["account_ids"]
    train_mask = data["train_mask"]

    feat_df_aligned = feat_df.reindex(saved_account_ids).fillna(0.0)
    raw_values = feat_df_aligned.values

    print(f"Aligned feature shape: {raw_values.shape} (should be [{len(saved_account_ids)}, 3])")
    print("\nRaw feature summary:")
    print(feat_df_aligned.describe())

    scaler = StandardScaler().fit(raw_values[train_mask.numpy()])
    scaled_values = scaler.transform(raw_values)
    joblib.dump(scaler, SCALER_OUTPUT_PATH)
    print(f"\nSaved scaler to {SCALER_OUTPUT_PATH} (fit on train accounts only).")

    torch.save({
        "features_raw": torch.tensor(raw_values, dtype=torch.float32),
        "features_scaled": torch.tensor(scaled_values, dtype=torch.float32),
        "feature_names": list(feat_df.columns),
    }, OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
