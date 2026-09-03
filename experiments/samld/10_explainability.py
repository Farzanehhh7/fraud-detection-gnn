import torch
import yaml

from fraud_gnn.models.graphsage_samld import StructuralOnlyGraphSAGE
from fraud_gnn.utils.metrics import load_checkpoint
from fraud_gnn.evaluation.explainability import explain_decision

CONFIG_PATH = "configs/samld_binary.yaml"
N_ACCOUNTS_TO_EXPLAIN = 3
DEFAULT_SEED = 42


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dict = torch.load(f"{cfg['output_prefix']}.pt", map_location="cpu", weights_only=False)
    x = data_dict["x"]
    edge_index = data_dict["edge_index"]
    y = data_dict["y_binary"]
    test_mask = data_dict["test_mask"]
    feature_names = data_dict.get("feature_cols", [f"feat_{i}" for i in range(x.shape[1])])

    model = StructuralOnlyGraphSAGE(in_channels=x.shape[1], hidden_channels=cfg["hidden_channels"],
                                     out_channels=cfg["out_channels"]).to(device)
    load_checkpoint(model, f"{cfg['checkpoint_dir']}/samld_seed_{DEFAULT_SEED}.pt", map_location=device)

    illicit_test_idx = (test_mask & (y == 1)).nonzero(as_tuple=True)[0]
    print(f"{len(illicit_test_idx)} illicit test accounts available to explain.")

    results = []
    for node_idx in illicit_test_idx[:N_ACCOUNTS_TO_EXPLAIN].tolist():
        result = explain_decision(model, x, edge_index, node_idx, feature_names, y, device)
        results.append(result)

    return results


if __name__ == "__main__":
    main()
