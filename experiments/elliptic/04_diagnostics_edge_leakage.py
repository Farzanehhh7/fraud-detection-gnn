import yaml

from fraud_gnn.data.elliptic import load_elliptic
from fraud_gnn.evaluation.diagnostics import check_edge_temporal_leakage

CONFIG_PATH = "configs/elliptic_binary.yaml"


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = load_elliptic(
        features_path=cfg["features_path"], edges_path=cfg["edges_path"], classes_path=cfg["classes_path"],
        train_end=cfg["train_end"], val_end=cfg["val_end"],
    )

    stats = check_edge_temporal_leakage(data["edge_index_cpu"], data["time_steps_raw"], cfg["train_end"])

    print(f"کل لبه‌ها: {stats['total_edges']}")
    print(f"لبه‌های رو به جلو (src_time <= dst_time): {stats['forward_edges']}")
    print(f"لبه‌های رو به عقب (src_time > dst_time): {stats['backward_edges']}")
    print(f"لبه‌های نشتی (مقصد در train، مبدأ از آینده): {stats['leaking_edges']}")
    print(f"تعداد لبه‌ها بعد از masking: {stats['n_edges_after_masking']}")

    return stats


if __name__ == "__main__":
    main()
