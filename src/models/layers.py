import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm, global_max_pool, JumpingKnowledge
from src.utils.config import load_all

model_cfg = load_all().get('model', {})
DEFAULT_IN = model_cfg.get('in_channels', 64)
DEFAULT_HIDDEN = model_cfg.get('hidden_dim', 256)
DEFAULT_HEADS = model_cfg.get('heads', 8)
DEFAULT_OUT = model_cfg.get('out_channels', 3)


class GATBackbone(nn.Module):
    """Two-layer GATv2 backbone with BatchNorm and JumpingKnowledge."""
    def __init__(self, in_channels=DEFAULT_IN, hidden_dim=DEFAULT_HIDDEN, heads=DEFAULT_HEADS):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_dim // heads, heads=heads)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads)
        self.bn2 = BatchNorm(hidden_dim)
        self.jk = JumpingKnowledge(mode='cat')

    def forward(self, x, edge_index):
        x1 = F.elu(self.bn1(self.conv1(x, edge_index)))
        x2 = F.elu(self.bn2(self.conv2(x1, edge_index)))
        return self.jk([x1, x2])


class Classifier(nn.Module):
    def __init__(self, hidden_dim=DEFAULT_HIDDEN, out_channels=DEFAULT_OUT):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, x, batch):
        return self.classifier(global_max_pool(x, batch))
"""Model layers."""
