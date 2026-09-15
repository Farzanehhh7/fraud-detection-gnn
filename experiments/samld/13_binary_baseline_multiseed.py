"""
پر کردن شکاف مستندشده در بخش ۷-۳-۴ گزارش پروژه: خط‌پایه‌ی بدون‌گراف برای
تشخیص دوتایی SAML-D (`01_tabular_baseline.py`) تنها یک‌بار، بدون تکرار seد،
اجرا شده بود. این اسکریپت همان دو مدل (LogisticRegression و RandomForest)
را روی پانزده seد رسمی پروتکل (بخش ۳-۷) تکرار می‌کند.

نکته‌ی عمدی نسبت به نسخه‌ی اولیه‌ی این اسکریپت: مقایسه‌ی آماری با GraphSAGE
اینجا انجام نمی‌شود (نیازی به بازاجرای 03_binary_training_multiseed.py
نیست) — چون اعداد ۱۵-seدی GraphSAGE از اجرای قبلی (۶ سپتامبر) از قبل در
دسترس‌اند. کافی‌ست خروجی این اسکریپت را کامل بفرستی؛ آزمون t جفتی را با
همان اعداد GraphSAGE موجود، جداگانه محاسبه می‌کنم.

اگر خط اول خروجی (کلیدهای samld_processed_v3.pt) چیزی غیر از "x"/"y" یا
"y_binary" برای برچسب یا ویژگی نشان داد، قبل از ادامه به من بگو تا
x_key/y_key پایین را اصلاح کنم.
"""
import os

import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from fraud_gnn.utils.metrics import evaluate_binary, find_best_threshold, run_multi_seed

OUTPUT_PREFIX = "samld_processed_v3"
SEEDS = (42, 1, 7, 123, 2024, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12)


def load_split():
    data_dict = torch.load(f"{OUTPUT_PREFIX}.pt", weights_only=False)
    print("کلیدهای موجود در samld_processed_v3.pt:", list(data_dict.keys()))

    x_key = "x"
    y_key = "y_binary" if "y_binary" in data_dict else ("y" if "y" in data_dict else None)

    if x_key not in data_dict or y_key is None:
        raise KeyError(
            "کلید مورد انتظار برای ویژگی ('x') یا برچسب ('y_binary'/'y') پیدا نشد. "
            "کلیدهای واقعی بالا چاپ شد — قبل از ادامه، این تابع را با نام درست اصلاح کن."
        )

    print(f"استفاده از x_key='{x_key}'، y_key='{y_key}'")

    x = data_dict[x_key].numpy()
    y = data_dict[y_key].numpy()
    train_mask = data_dict["train_mask"].numpy()
    val_mask = data_dict["val_mask"].numpy()
    test_mask = data_dict["test_mask"].numpy()

    return (x[train_mask], y[train_mask], x[val_mask], y[val_mask], x[test_mask], y[test_mask])


def run_one_seed(seed, model_fn, X_train, y_train, X_val, y_val, X_test, y_test, name):
    model = model_fn(seed)
    model.fit(X_train, y_train)

    probs_val = model.predict_proba(X_val)[:, 1]
    best_t, _ = find_best_threshold(y_val, probs_val)

    probs_test = model.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= best_t).astype(int)
    metrics = evaluate_binary(name, y_test, preds_test, probs_test, verbose=False)
    metrics["threshold"] = best_t
    return metrics


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_split()
    print(f"Train: {len(X_train)}   Val: {len(X_val)}   Test: {len(X_test)}")
    print(f"illicit در train: {int(y_train.sum())}   در val: {int(y_val.sum())}   در test: {int(y_test.sum())}")

    os.makedirs("outputs/metrics", exist_ok=True)

    print("\n=== Logistic Regression بدون گراف — پانزده seed ===")
    print("(نکته: LR با solver پیش‌فرض غالباً قطعی است؛ اگر انحراف‌معیار صفر شد، این خطا نیست.)")
    df_lr, summary_lr = run_multi_seed(
        lambda seed: run_one_seed(
            seed, lambda s: LogisticRegression(class_weight="balanced", max_iter=1000, random_state=s),
            X_train, y_train, X_val, y_val, X_test, y_test, "LR بدون گراف",
        ),
        seeds=SEEDS, name="LR بدون گراف (۱۵ seed)",
    )
    df_lr.to_csv("outputs/metrics/samld_lr_nograph_15seed_results.csv", index=False)
    print("ذخیره شد: outputs/metrics/samld_lr_nograph_15seed_results.csv")

    print("\n=== Random Forest بدون گراف — پانزده seed ===")
    df_rf, summary_rf = run_multi_seed(
        lambda seed: run_one_seed(
            seed, lambda s: RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=s, n_jobs=-1,
            ),
            X_train, y_train, X_val, y_val, X_test, y_test, "RF بدون گراف",
        ),
        seeds=SEEDS, name="RF بدون گراف (۱۵ seed)",
    )
    df_rf.to_csv("outputs/metrics/samld_rf_nograph_15seed_results.csv", index=False)
    print("ذخیره شد: outputs/metrics/samld_rf_nograph_15seed_results.csv")

    return df_lr, summary_lr, df_rf, summary_rf


if __name__ == "__main__":
    main()
