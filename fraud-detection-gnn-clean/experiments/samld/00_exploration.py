import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import yaml

from fraud_gnn.utils.seeding import set_seed

CONFIG_PATH = "configs/samld_binary.yaml"


def explore_and_build_graph(file_path, n_rows=200000):
    print(f"--- Phase 1: Exploration (First {n_rows} rows) ---")
    df = pd.read_csv(file_path, nrows=n_rows)

    target_col = "Is_laundering"
    type_col = "Laundering_type"
    sender_col = "Sender_account"
    receiver_col = "Receiver_account"

    print("\nTop 10 Money Laundering Typologies:")
    print(df[type_col].value_counts().head(10))

    print("\nLaundering vs Normal Transactions:")
    print(df[target_col].value_counts())

    print("\n--- Phase 2: Graph Stats ---")
    all_accounts = np.unique(df[[sender_col, receiver_col]].values)

    print(f"Total Nodes (Accounts): {len(all_accounts)}")
    print(f"Total Edges (Transactions): {len(df)}")

    le = LabelEncoder()
    df[type_col] = df[type_col].fillna("Normal")
    labels = le.fit_transform(df[type_col])
    print(f"Number of unique laundering patterns: {len(le.classes_)}")
    print("Sample of classes:", le.classes_[:5])

    return True


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(42)
    return explore_and_build_graph(cfg["file_path"])


if __name__ == "__main__":
    main()
