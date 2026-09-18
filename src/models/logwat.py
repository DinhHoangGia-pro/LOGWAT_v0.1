import torch.nn as nn
from src.models.layers import GATBackbone, Classifier, DEFAULT_IN, DEFAULT_HIDDEN, DEFAULT_OUT


class HeavyWebGNN(nn.Module):
    def __init__(self, in_channels=DEFAULT_IN, hidden_dim=DEFAULT_HIDDEN, out_channels=DEFAULT_OUT,
                 use_edge_attr=False, edge_dim=3):
        super().__init__()
        self.backbone = GATBackbone(in_channels=in_channels, hidden_dim=hidden_dim,
                                     use_edge_attr=use_edge_attr, edge_dim=edge_dim)
        self.classifier = Classifier(hidden_dim=hidden_dim, out_channels=out_channels)

    def forward(self, x, edge_index, batch, edge_attr=None):
        feats = self.backbone(x, edge_index, edge_attr=edge_attr)
        return self.classifier(feats, batch)
"""LOGWAT model entrypoint."""

class LOGWATModel:
    pass
