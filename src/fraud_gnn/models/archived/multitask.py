import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class MultiTaskGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_type_classes):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.binary_head = nn.Linear(hidden_channels, 2)
        self.type_head = nn.Linear(hidden_channels, num_type_classes)

    def forward(self, x, edge_index):
        h = F.dropout(F.relu(self.conv1(x, edge_index)), p=0.3, training=self.training)
        h = F.dropout(F.relu(self.conv2(h, edge_index)), p=0.3, training=self.training)
        return self.binary_head(h), self.type_head(h)
