import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score

from fraud_gnn.data.graph_utils import build_adjacency, build_batch_subgraph
from fraud_gnn.models.graphsage_samld import StructuralOnlyGraphSAGE
from fraud_gnn.utils.metrics import report_at_percentile_thresholds, bootstrap_test_ci
from fraud_gnn.utils.seeding import set_seed

CONFIG_PATH = "configs/samld_binary.yaml"
BATCH_SIZE = 256
WEIGHT_CAP = 30.0
EPOCHS = 20
SEED = 42


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(SEED)
    random.seed(SEED)

    print("در حال بارگذاری داده پردازش‌شده SAML-D...")
    data_dict = torch.load(f"{cfg['output_prefix']}.pt", weights_only=False)
    x = data_dict["x"]
    edge_index = data_dict["edge_index"]
    y = data_dict["y_binary"]
    train_mask = data_dict["train_mask"]
    test_mask = data_dict["test_mask"]

    n_pos = (y[train_mask] == 1).sum().item()
    n_neg = (y[train_mask] == 0).sum().item()
    class_weight = torch.tensor([1.0, min(n_neg / max(n_pos, 1), WEIGHT_CAP)])

    adjacency = build_adjacency(edge_index, x.shape[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StructuralOnlyGraphSAGE(in_channels=x.shape[1], hidden_channels=cfg["hidden_channels"],
                                     out_channels=cfg["out_channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weight.to(device))

    train_node_ids = train_mask.nonzero(as_tuple=True)[0].tolist()

    print("در حال آموزش...")
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

    print("آموزش تمام شد.\n")

    model.eval()
    with torch.no_grad():
        out = model(x.to(device), edge_index.to(device))
        probs_all = F.softmax(out, dim=1)[:, 1].cpu().numpy()

    y_test = y[test_mask].numpy()
    probs_test = probs_all[test_mask.numpy()]

    print("=== گزارش threshold صدک‌محور روی test ===")
    report_at_percentile_thresholds(y_test, probs_test, percentiles=(90, 95, 99))

    print("\n=== فاصله اطمینان bootstrap برای PR-AUC-محور، مستقل از threshold ===")
    pr_auc = average_precision_score(y_test, probs_test)
    print(f"PR-AUC روی test: {pr_auc:.4f}")
    bootstrap_test_ci(y_test, probs_test, n_iterations=100, sample_frac=0.5, threshold=0.5)


if __name__ == "__main__":
    main()
