import torch
import torch.nn as nn


class TIEBaseline(nn.Module):
    def __init__(self, in_channels, num_timesteps, temporal_dim, hidden_channels, out_channels):
        super().__init__()
        self.temporal_embed = nn.Embedding(num_timesteps, temporal_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + temporal_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, time_steps):
        t_emb = self.temporal_embed(time_steps)
        combined = torch.cat([x, t_emb], dim=1)
        return self.mlp(combined)
