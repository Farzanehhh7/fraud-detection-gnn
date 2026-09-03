import yaml

from fraud_gnn.data.samld import build_samld_graph

CONFIG_PATH = "configs/samld_binary.yaml"


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    result = build_samld_graph(
        file_path=cfg["file_path"], n_rows=cfg["n_rows"], output_prefix=cfg["output_prefix"], save=True,
    )

    print(f"Nodes: {result['x'].shape[0]}   Features: {result['x'].shape[1]}")
    print(f"Train: {result['train_mask'].sum().item()}   "
          f"Val: {result['val_mask'].sum().item()}   Test: {result['test_mask'].sum().item()}")
    print(f"Types: {result['num_types']}")

    return result


if __name__ == "__main__":
    main()
