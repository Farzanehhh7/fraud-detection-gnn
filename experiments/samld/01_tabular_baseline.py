import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from fraud_gnn.utils.metrics import evaluate_binary, find_best_threshold
from fraud_gnn.utils.seeding import set_seed

CONFIG_PATH = "configs/samld_binary.yaml"


def get_feature_cols(df):
    candidates = [
        "Amount", "Payment_currency", "Received_currency",
        "Sender_bank_location", "Receiver_bank_location",
        "Payment_type", "cross_currency", "cross_border",
    ]
    return [c for c in candidates if c in df.columns]


def load_and_prepare_raw(file_path, n_rows):
    print(f"Loading {n_rows} rows from SAML-D...")
    df = pd.read_csv(file_path, nrows=n_rows)

    cat_cols = ["Payment_currency", "Received_currency",
                "Sender_bank_location", "Receiver_bank_location", "Payment_type"]
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    if "Payment_currency" in df.columns and "Received_currency" in df.columns:
        df["cross_currency"] = (df["Payment_currency"] != df["Received_currency"]).astype(int)
    if "Sender_bank_location" in df.columns and "Receiver_bank_location" in df.columns:
        df["cross_border"] = (df["Sender_bank_location"] != df["Receiver_bank_location"]).astype(int)

    return df


def task_a_binary_xgboost(df):
    print("\n" + "=" * 60 + "\nTask A - Binary (XGBoost, raw transaction features)\n" + "=" * 60)
    feature_cols = get_feature_cols(df)
    X, y = df[feature_cols], df["Is_laundering"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                           scale_pos_weight=n_neg / n_pos, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    evaluate_binary("XGBoost - Is Laundering", y_test, y_pred, y_prob)


def task_b_type_xgboost(df):
    print("\n" + "=" * 60 + "\nTask B - Multi-class type (XGBoost, illicit only)\n" + "=" * 60)
    df_illicit = df[df["Is_laundering"] == 1].copy()
    print(f"تعداد نمونه‌های واقعا پول‌شویی: {len(df_illicit)}")

    feature_cols = get_feature_cols(df_illicit)
    le_target = LabelEncoder()
    y = le_target.fit_transform(df_illicit["Laundering_type"])
    X = df_illicit[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    model = XGBClassifier(n_estimators=100, objective="multi:softprob",
                           num_class=len(le_target.classes_), tree_method="hist", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, labels=np.arange(len(le_target.classes_)),
                                 target_names=le_target.classes_, zero_division=0))


def run_and_report(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)
    probs_val = model.predict_proba(X_val)[:, 1]
    best_t, _ = find_best_threshold(y_val, probs_val)

    probs_test = model.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= best_t).astype(int)
    evaluate_binary(name, y_test, preds_test, probs_test)
    print(f"threshold انتخابی: {best_t:.2f}")


def binary_lr_rf_on_processed_features(output_prefix):
    print("در حال بارگذاری داده پردازش‌شده SAML-D...")
    data_dict = torch.load(f"{output_prefix}.pt", weights_only=False)
    x = data_dict["x"].numpy()
    y = data_dict["y_binary"].numpy()
    train_mask = data_dict["train_mask"].numpy()
    val_mask = data_dict["val_mask"].numpy()
    test_mask = data_dict["test_mask"].numpy()

    X_train, y_train = x[train_mask], y[train_mask]
    X_val, y_val = x[val_mask], y[val_mask]
    X_test, y_test = x[test_mask], y[test_mask]

    print(f"Train: {len(X_train)}   Val: {len(X_val)}   Test: {len(X_test)}")
    print(f"illicit در train: {y_train.sum()}   در val: {y_val.sum()}   در test: {y_test.sum()}")

    print("\n=== Logistic Regression، بدون گراف ===")
    run_and_report("LR بدون گراف", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
                    X_train, y_train, X_val, y_val, X_test, y_test)

    print("\n=== Random Forest، بدون گراف ===")
    run_and_report("RF بدون گراف",
                    RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
                    X_train, y_train, X_val, y_val, X_test, y_test)


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(42)

    df = load_and_prepare_raw(cfg["file_path"], n_rows=500000)
    task_a_binary_xgboost(df)
    task_b_type_xgboost(df)

    binary_lr_rf_on_processed_features(cfg["output_prefix"])


if __name__ == "__main__":
    main()
