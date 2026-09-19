"""Graph-based baselines (Group A of the baseline-fairness rerun, #16).

Each baseline is `GATBackbone`'s exact skeleton -- 2 conv layers at the same
hidden_dim, BatchNorm, ELU, JumpingKnowledge('cat'), and the SAME `Classifier`
head (global max-pool -> MLP) -- with ONLY the convolution operator swapped.
Same input graphs (web_graphs.pkl: same node features, same E_seq+E_skip+E_sem
edge_index, same split), so any difference in accuracy is attributable to the
learning architecture, not the graph representation or the classifier head.

All models share the interface `forward(x, edge_index, batch, edge_attr=None)`
that `src.training.train.train()` and the evaluation scripts already call.

Only `HGTBaseline` looks at `edge_attr`: it needs the relation type of each edge
(the 3-dim one-hot seq/skip/sem produced by
`build_single_graph(..., use_edge_attr=True)`) to form typed edges. GCN, SAGE and
GIN are relation-agnostic by construction and ignore it -- they see E_seq+E_skip+E_sem
as one merged edge set, which is precisely what "single-relation graph" means here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GCNConv, GINConv, HGTConv, JumpingKnowledge, SAGEConv

from src.bag.graph_builder import RELATION_TYPES
from src.models.layers import Classifier, DEFAULT_HIDDEN, DEFAULT_IN, DEFAULT_OUT

NODE_TYPE = 'token'


class _ConvBaseline(nn.Module):
    """Two conv layers + BN + ELU + JK('cat') + shared Classifier."""

    needs_edge_attr = False

    def __init__(self, conv1, conv2, hidden_dim, out_channels):
        super().__init__()
        self.conv1 = conv1
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = conv2
        self.bn2 = BatchNorm(hidden_dim)
        self.jk = JumpingKnowledge(mode='cat')
        self.classifier = Classifier(hidden_dim=hidden_dim, out_channels=out_channels)

    def _convs(self, x, edge_index, edge_attr, conv):
        return conv(x, edge_index)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x1 = F.elu(self.bn1(self._convs(x, edge_index, edge_attr, self.conv1)))
        x2 = F.elu(self.bn2(self._convs(x1, edge_index, edge_attr, self.conv2)))
        return self.classifier(self.jk([x1, x2]), batch)


class GCNBaseline(_ConvBaseline):
    def __init__(self, in_channels=DEFAULT_IN, hidden_dim=DEFAULT_HIDDEN, out_channels=DEFAULT_OUT):
        super().__init__(GCNConv(in_channels, hidden_dim), GCNConv(hidden_dim, hidden_dim),
                         hidden_dim, out_channels)


class SAGEBaseline(_ConvBaseline):
    def __init__(self, in_channels=DEFAULT_IN, hidden_dim=DEFAULT_HIDDEN, out_channels=DEFAULT_OUT):
        super().__init__(SAGEConv(in_channels, hidden_dim), SAGEConv(hidden_dim, hidden_dim),
                         hidden_dim, out_channels)


def _gin_mlp(in_dim, hidden_dim):
    """The standard GIN update network: a 2-layer MLP."""
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))


class GINBaseline(_ConvBaseline):
    def __init__(self, in_channels=DEFAULT_IN, hidden_dim=DEFAULT_HIDDEN, out_channels=DEFAULT_OUT):
        super().__init__(GINConv(_gin_mlp(in_channels, hidden_dim), train_eps=True),
                         GINConv(_gin_mlp(hidden_dim, hidden_dim), train_eps=True),
                         hidden_dim, out_channels)


class HGTBaseline(_ConvBaseline):
    """HGTConv over ONE node type ('token') and THREE edge types (seq/skip/sem).

    Edge types are recovered from the one-hot `edge_attr` (argmax over the
    RELATION_TYPES columns), so this must be trained/evaluated on graphs built with
    `use_edge_attr=True`. Every edge type is always present in the dict (possibly
    empty) so the parameter set never depends on which relations a batch contains.
    """

    needs_edge_attr = True

    def __init__(self, in_channels=DEFAULT_IN, hidden_dim=DEFAULT_HIDDEN, out_channels=DEFAULT_OUT,
                 heads=8):
        edge_types = [(NODE_TYPE, r, NODE_TYPE) for r in RELATION_TYPES]
        metadata = ([NODE_TYPE], edge_types)
        super().__init__(HGTConv(in_channels, hidden_dim, metadata, heads=heads),
                         HGTConv(hidden_dim, hidden_dim, metadata, heads=heads),
                         hidden_dim, out_channels)
        self.edge_types = edge_types

    def _convs(self, x, edge_index, edge_attr, conv):
        if edge_attr is None:
            raise ValueError("HGTBaseline needs edge_attr (relation one-hot): build graphs with use_edge_attr=True")
        rel = edge_attr.argmax(dim=1)
        edge_index_dict = {et: edge_index[:, rel == i] for i, et in enumerate(self.edge_types)}
        return conv({NODE_TYPE: x}, edge_index_dict)[NODE_TYPE]


GRAPH_BASELINES = {
    'gcn': GCNBaseline,
    'graphsage': SAGEBaseline,
    'gin': GINBaseline,
    'hgt': HGTBaseline,
}
