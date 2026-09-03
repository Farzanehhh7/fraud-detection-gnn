import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv


class GATFraudDetector(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=0.2)
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def compute_global_context(x, time_steps, num_timesteps):
    feat_dim = x.size(1)
    ctx_sum = torch.zeros(num_timesteps, feat_dim, device=x.device)
    counts = torch.zeros(num_timesteps, device=x.device)
    ctx_sum.index_add_(0, time_steps, x)
    counts.index_add_(0, time_steps, torch.ones_like(time_steps, dtype=torch.float))
    counts = counts.clamp(min=1).unsqueeze(1)
    timestep_mean = ctx_sum / counts
    deviation = x - timestep_mean[time_steps]
    return deviation


class TemporalEmbedding(nn.Module):
    def __init__(self, num_timesteps, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_timesteps, embed_dim)

    def forward(self, time_steps):
        return self.embedding(time_steps)


class TripleAttentionLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, temporal_dim, global_in_channels,
                 use_temporal=True, use_global=True):
        super().__init__()
        self.use_temporal = use_temporal
        self.use_global = use_global
        self.struct_conv = SAGEConv(in_channels, hidden_channels)
        n_streams = 1 + int(use_temporal) + int(use_global)
        if use_temporal:
            self.temporal_proj = nn.Sequential(nn.Linear(temporal_dim, hidden_channels), nn.ReLU())
        if use_global:
            self.global_proj = nn.Sequential(nn.Linear(global_in_channels, hidden_channels), nn.ReLU())
        self.attn_gate = nn.Linear(hidden_channels * n_streams, n_streams)

    def forward(self, x, edge_index, temporal_emb, global_ctx_per_node):
        streams = [F.dropout(self.struct_conv(x, edge_index), p=0.2, training=self.training)]
        if self.use_temporal:
            streams.append(self.temporal_proj(temporal_emb))
        if self.use_global:
            streams.append(self.global_proj(global_ctx_per_node))

        combined = torch.cat(streams, dim=1)
        weights = F.softmax(self.attn_gate(combined), dim=1)
        out = sum(weights[:, i:i + 1] * streams[i] for i in range(len(streams)))
        return out, weights


class ATGATGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_timesteps,
                 temporal_dim=32, use_temporal=True, use_global=True):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.use_temporal = use_temporal
        self.use_global = use_global
        self.temporal_embed = TemporalEmbedding(num_timesteps, temporal_dim)

        self.layer1 = TripleAttentionLayer(in_channels, hidden_channels, temporal_dim,
                                            in_channels, use_temporal, use_global)
        self.layer2 = TripleAttentionLayer(hidden_channels, hidden_channels, temporal_dim,
                                            hidden_channels, use_temporal, use_global)

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, time_steps):
        t_emb = self.temporal_embed(time_steps) if self.use_temporal else None

        gc1 = compute_global_context(x, time_steps, self.num_timesteps) if self.use_global else None
        h, w1 = self.layer1(x, edge_index, t_emb, gc1)
        h = self.dropout(F.relu(h))

        gc2 = compute_global_context(h, time_steps, self.num_timesteps) if self.use_global else None
        h, w2 = self.layer2(h, edge_index, t_emb, gc2)
        h = self.dropout(F.relu(h))

        out = self.classifier(h)
        return out, (w1, w2)


def compute_global_deviation(x, time_steps, num_timesteps):
    feat_dim = x.size(1)
    ctx_sum = torch.zeros(num_timesteps, feat_dim, device=x.device)
    counts = torch.zeros(num_timesteps, device=x.device)
    ctx_sum.index_add_(0, time_steps, x)
    counts.index_add_(0, time_steps, torch.ones_like(time_steps, dtype=torch.float))
    counts = counts.clamp(min=1).unsqueeze(1)
    timestep_mean = ctx_sum / counts
    return x - timestep_mean[time_steps]


class ModulatedLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, temporal_dim, global_in_channels):
        super().__init__()
        self.struct_conv = SAGEConv(in_channels, hidden_channels)
        self.temporal_gamma = nn.Linear(temporal_dim, hidden_channels)
        self.temporal_beta = nn.Linear(temporal_dim, hidden_channels)
        self.global_proj = nn.Sequential(nn.Linear(global_in_channels, hidden_channels), nn.ReLU())

    def forward(self, x, edge_index, temporal_emb, global_dev):
        h_struct = F.dropout(self.struct_conv(x, edge_index), p=0.2, training=self.training)
        gamma = torch.tanh(self.temporal_gamma(temporal_emb))
        beta = self.temporal_beta(temporal_emb)
        h_modulated = h_struct * (1.0 + gamma) + beta
        h_global = self.global_proj(global_dev)
        return h_modulated + h_global


class ATGATModulated(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_timesteps, temporal_dim=32):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.temporal_embed = TemporalEmbedding(num_timesteps, temporal_dim)
        self.layer1 = ModulatedLayer(in_channels, hidden_channels, temporal_dim, in_channels)
        self.layer2 = ModulatedLayer(hidden_channels, hidden_channels, temporal_dim, hidden_channels)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, time_steps):
        t_emb = self.temporal_embed(time_steps)

        gd1 = compute_global_deviation(x, time_steps, self.num_timesteps)
        h = self.layer1(x, edge_index, t_emb, gd1)
        h = self.dropout(F.relu(h))

        gd2 = compute_global_deviation(h, time_steps, self.num_timesteps)
        h = self.layer2(h, edge_index, t_emb, gd2)
        h = self.dropout(F.relu(h))

        return self.classifier(h)
