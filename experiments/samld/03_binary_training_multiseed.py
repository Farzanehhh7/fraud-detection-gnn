import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from fraud_gnn.data.graph_utils import build_adjacency, build_batch_subgraph
from fraud_gnn.models.graphsage_samld import StructuralOnlyGraphSAGE
from fraud_gnn.utils.metrics import evaluate_binary, find_best_threshold, run_multi_seed, save_checkpoint
from fraud_gnn.utils.seeding import set_seed
from fraud_gnn.utils.run_logger import log_run

CONFIG_PATH = "configs/samld_binary.yaml"
BATCH_SIZE = 256
WEIGHT_CAP = 30.0
EPOCHS = 20


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dict = torch.load(f"{cfg['output_prefix']}.pt", weights_only=False)
    x = data_dict["x"]
    edge_index = data_dict["edge_index"]
    y = data_dict["y_binary"]
    train_mask = data_dict["train_mask"]
    val_mask = data_dict["val_mask"]
    test_mask = data_dict["test_mask"]

    n_pos = (y[train_mask] == 1).sum().item()
    n_neg = (y[train_mask] == 0).sum().item()
    raw_weight = n_neg / max(n_pos, 1)
    class_weight = torch.tensor([1.0, min(raw_weight, WEIGHT_CAP)])

    adjacency = build_adjacency(edge_index, x.shape[0])
    train_node_ids_master = train_mask.nonzero(as_tuple=True)[0].tolist()

    def run_one_seed(seed):
        set_seed(seed)
        random.seed(seed)
        train_node_ids = list(train_node_ids_master)

        model = StructuralOnlyGraphSAGE(
            in_channels=x.shape[1], hidden_channels=cfg["hidden_channels"], out_channels=cfg["out_channels"],
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss(weight=class_weight.to(device))

        for epoch in range(1, EPOCHS + 1):
            random.shuffle(train_node_ids)
            model.train()
            for i in range(0, len(train_node_ids), BATCH_SIZE):
                seed_nodes = train_node_ids[i:i + BATCH_SIZE]
                sub_x, sub_edge_index, seed_local_idx = build_batch_subgraph(seed_nodes, adjacency, x)
                sub_x, sub_edge_index = sub_x.to(device), sub_edge_index.to(device)
                seed_local_idx = seed_local_idx.to(device)
                batch_y = y[seed_nodes].to(device)

                optimizer.zero_grad()
                out = model(sub_x, sub_edge_index)
                loss = criterion(out[seed_local_idx], batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(x.to(device), edge_index.to(device))
            probs_all = F.softmax(out, dim=1)[:, 1].cpu().numpy()

        y_val = y[val_mask].numpy()
        probs_val = probs_all[val_mask.numpy()]
        best_t, _ = find_best_threshold(y_val, probs_val)

        y_test = y[test_mask].numpy()
        probs_test = probs_all[test_mask.numpy()]
        preds_test = (probs_test >= best_t).astype(int)

        metrics = evaluate_binary("SAML-D GraphSAGE", y_test, preds_test, probs_test, verbose=False)
        metrics["threshold"] = best_t

        checkpoint_path = f"{cfg['checkpoint_dir']}/samld_seed_{seed}.pt"
        save_checkpoint(
            model, checkpoint_path,
            extra={"seed": seed, "F1": metrics["F1"], "threshold": float(best_t),
                   "hidden_channels": cfg["hidden_channels"], "lr": 0.005, "in_channels": x.shape[1]},
        )

        log_run(
            phase="samld", experiment="03_binary_training_multiseed", seed=seed,
            metric_name="F1", metric_value=metrics["F1"],
            checkpoint_path=checkpoint_path, log_path="outputs/logs/samld_03.log",
        )

        return metrics

    df, summary = run_multi_seed(run_one_seed, seeds=cfg["seeds"], name="SAML-D GraphSAGE binary")
    return df, summary


if __name__ == "__main__":
    main()
