import torch
import yaml

from fraud_gnn.data.elliptic import load_elliptic
from fraud_gnn.models.graphsage_elliptic import StructuralOnlyGraphSAGE
from fraud_gnn.utils.metrics import load_checkpoint
from fraud_gnn.evaluation.explainability import explain_decision

CONFIG_PATH = "configs/elliptic_binary.yaml"
CHECKPOINT_PATH = "structural_only_best.pt"
N_ACCOUNTS_TO_EXPLAIN = 3


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_elliptic(
        features_path=cfg["features_path"], edges_path=cfg["edges_path"], classes_path=cfg["classes_path"],
        train_end=cfg["train_end"], val_end=cfg["val_end"], device=device,
    )
    x, y, edge_index = data["x"], data["y"], data["edge_index"]
    test_mask = data["test_mask"]
    feature_names = [f"feat_{i}" for i in range(x.shape[1])]

    model = StructuralOnlyGraphSAGE(in_channels=x.shape[1], hidden_channels=cfg["hidden_channels"],
                                     out_channels=cfg["out_channels"]).to(device)
    load_checkpoint(model, CHECKPOINT_PATH, map_location=device)

    illicit_test_idx = (test_mask & (y == 1)).nonzero(as_tuple=True)[0]
    print(f"{len(illicit_test_idx)} illicit test accounts available to explain.")

    results = []
    for node_idx in illicit_test_idx[:N_ACCOUNTS_TO_EXPLAIN].tolist():
        result = explain_decision(model, x, edge_index, node_idx, feature_names, y, device)
        results.append(result)

    return results


if __name__ == "__main__":
    main()
