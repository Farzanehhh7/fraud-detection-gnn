import yaml

from fraud_gnn.data.elliptic import load_elliptic
from fraud_gnn.evaluation.diagnostics import compute_homophily

CONFIG_PATH = "configs/elliptic_binary.yaml"


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = load_elliptic(
        features_path=cfg["features_path"], edges_path=cfg["edges_path"], classes_path=cfg["classes_path"],
        train_end=cfg["train_end"], val_end=cfg["val_end"],
    )
    df_class = data["df_class"]
    label_map = dict(zip(df_class["txId"], df_class["label"]))

    stats = compute_homophily(data["df_edge"], label_map)

    print(f"لبه‌های با هر دو سر برچسب‌دار: {stats['n_both_labeled']}")
    print(f"همگنی لبه‌ای (edge homophily): {stats['edge_homophily']:.4f}")
    print(f"جفت‌های illicit-illicit: {stats['illicit_pairs']}")
    print(f"جفت‌های licit-licit: {stats['licit_pairs']}")
    print(f"جفت‌های مخلوط: {stats['mixed_pairs']}")
    print(f"\nهمگنی گره‌ای (میانگین کل): {stats['node_homophily_all']:.4f}")
    print(f"همگنی گره‌ای (فقط illicit، n={stats['n_illicit_nodes']}): {stats['node_homophily_illicit']:.4f}")
    print(f"همگنی گره‌ای (فقط licit، n={stats['n_licit_nodes']}): {stats['node_homophily_licit']:.4f}")

    return stats


if __name__ == "__main__":
    main()
