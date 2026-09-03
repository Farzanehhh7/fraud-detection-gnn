import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SAGEBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = SAGEConv(in_channels, hidden_channels)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        return F.dropout(h, p=0.2, training=self.training)


class StructuralOnlyGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.block1 = SAGEBlock(in_channels, hidden_channels)
        self.block2 = SAGEBlock(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, return_embeddings=False):
        h1 = self.dropout(F.relu(self.block1(x, edge_index)))
        h2 = self.dropout(F.relu(self.block2(h1, edge_index)))
        out = self.classifier(h2)
        if return_embeddings:
            return out, h1, h2
        return out
