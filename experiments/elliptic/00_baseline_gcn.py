import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
import yaml

from fraud_gnn.utils.metrics import evaluate_gnn, build_edge_index
from fraud_gnn.utils.seeding import set_seed

CONFIG_PATH = "configs/elliptic_binary.yaml"


class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(42)

    df_feat = pd.read_csv(cfg["features_path"], header=None)
    df_edge = pd.read_csv(cfg["edges_path"])
    df_class = pd.read_csv(cfg["classes_path"])

    df_feat.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(165)]
    df_class.columns = ["txId", "class"]

    _, edge_index = build_edge_index(df_feat["txId"].values, df_edge["txId1"], df_edge["txId2"])

    x = df_feat.drop(columns=["txId", "time_step"]).values
    scaler = StandardScaler()
    x = torch.tensor(scaler.fit_transform(x), dtype=torch.float)

    df_class["label"] = df_class["class"].map({"1": 1, "2": 0, "unknown": -1})
    y = torch.tensor(df_class["label"].values, dtype=torch.long)

    train_idx = df_feat[df_feat["time_step"] <= 34].index
    test_idx = df_feat[df_feat["time_step"] > 34].index

    train_mask = (y != -1) & torch.zeros(len(y), dtype=torch.bool).scatter_(0, torch.tensor(train_idx), True)
    test_mask = (y != -1) & torch.zeros(len(y), dtype=torch.bool).scatter_(0, torch.tensor(test_idx), True)

    n_pos = (y[train_mask] == 1).sum().item()
    n_neg = (y[train_mask] == 0).sum().item()
    weight = torch.tensor([1.0, n_neg / n_pos])
    print(f"Class Weight (illicit): {n_neg / n_pos:.2f}")

    model = SimpleGCN(in_channels=165, hidden_channels=128, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    print("\nStarting Training...")
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    return evaluate_gnn("GCN Simple (Final)", model, x, edge_index, y, test_mask, two_class_softmax=True)


if __name__ == "__main__":
    main()
