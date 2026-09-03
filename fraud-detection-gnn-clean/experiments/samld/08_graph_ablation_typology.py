import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from fraud_gnn.models.graphsage_samld import merge_rare_types

CONFIG_PATH = "configs/samld_binary.yaml"
TYPOLOGY_CONFIG_PATH = "configs/samld_typology.yaml"
REGISTRY_PATH = "outputs/run_registry.csv"


def fit_and_evaluate_no_graph(x, y_type_remapped, train_mask, test_mask, seed):
    illicit_train = train_mask & (y_type_remapped != -1)
    illicit_test = test_mask & (y_type_remapped != -1)

    X_train = x[illicit_train].numpy()
    y_train = y_type_remapped[illicit_train].numpy()
    X_test = x[illicit_test].numpy()
    y_test = y_type_remapped[illicit_test].numpy()

    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }


def load_from_registry(experiment, metric_name, seeds):
    registry = pd.read_csv(REGISTRY_PATH)
    subset = registry[(registry["experiment"] == experiment) & (registry["metric_name"] == metric_name)]
    subset = subset.drop_duplicates(subset="seed", keep="last").set_index("seed")
    values = []
    for seed in seeds:
        if seed not in subset.index:
            return None
        values.append(subset.loc[seed, "metric_value"])
    return np.array(values, dtype=float)


def summarize(vals, label):
    print(f"{label:45s} macro_f1 = {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")
    return vals


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(TYPOLOGY_CONFIG_PATH, encoding="utf-8") as f:
        typ_cfg = yaml.safe_load(f)

    seeds = cfg["seeds"]

    print("Loading real SAML-D v3 data...")
    data_dict = torch.load(f"{cfg['output_prefix']}.pt", map_location="cpu", weights_only=False)
    x = data_dict["x"]
    y_type = data_dict["y_type"]
    train_mask = data_dict["train_mask"]
    test_mask = data_dict["test_mask"]

    remap, num_new_types, _ = merge_rare_types(
        y_type, train_mask, min_train_samples=typ_cfg["min_train_samples_per_type"],
    )
    y_type_remapped = y_type.clone()
    for old_id, new_id in remap.items():
        y_type_remapped[y_type == old_id] = new_id

    print(f"\n=== NO-GRAPH type classifier: raw 8 account features only ({len(seeds)} seeds) ===")
    no_graph_metrics = []
    for seed in seeds:
        m = fit_and_evaluate_no_graph(x, y_type_remapped, train_mask, test_mask, seed)
        no_graph_metrics.append(m)
        print(f"  seed={seed}  accuracy={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")

    print(f"\n{'=' * 70}\n=== FULL LADDER ({len(seeds)} seeds): how much does each graph ingredient add? ===\n{'=' * 70}")

    no_graph_vals = summarize(np.array([m["macro_f1"] for m in no_graph_metrics]), "NO GRAPH (raw 8 features only)")

    embedding_only_vals = load_from_registry("06_typology_two_stage", "macro_f1_embedding_only", seeds)
    embedding_structural_vals = load_from_registry("06_typology_two_stage", "macro_f1_embedding_structural", seeds)
    multitask_vals = load_from_registry("archived_multitask_variant", "macro_f1_type", seeds)

    ladder = [
        ("embedding_only", embedding_only_vals, "GNN embedding only (06, ثبت‌شده در registry)"),
        ("embedding+structural", embedding_structural_vals, "GNN embedding + structural (06، ثبت‌شده در registry)"),
        ("multi_task", multitask_vals, "Multi-task shared backbone (archived، ثبت‌شده در registry)"),
    ]

    for _, vals, label in ladder:
        if vals is not None:
            summarize(vals, label)
        else:
            print(f"{label:45s} در run_registry.csv پیدا نشد — اول تجربه‌ی مربوطه رو اجرا کن.")

    print(f"\n--- Paired t-tests against NO GRAPH (df={len(seeds) - 1}) ---")
    for name, vals, _ in ladder:
        if vals is None:
            continue
        t, p = stats.ttest_rel(vals, no_graph_vals)
        delta = vals.mean() - no_graph_vals.mean()
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"  {name:25s} delta={delta:+.4f}  t={t:.3f}  p={p:.4f}  ({sig})")


if __name__ == "__main__":
    main()
