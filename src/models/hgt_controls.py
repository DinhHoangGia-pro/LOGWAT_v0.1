"""Control variants of `HGTBaseline` for isolating WHAT makes HGT pass the held-out matrix.

Same skeleton as `HGTBaseline` (2 x HGTConv, 8 heads, hidden 256, BatchNorm, ELU,
JumpingKnowledge, shared Classifier); ONLY the edge-type assignment differs:

  collapsed  every edge gets ONE edge type -> HGTConv keeps its dot-product attention,
             separate K/Q/V, GELU+out_lin and skip gate, but has no relation information
             (equivalent to a homogeneous graph). If this still solves comment_splitting,
             the edge TYPES are not what solves it.
  random     three edge types, but assigned by a fixed hash of (src, dst) node index, so the
             parameter count / per-type transforms equal the real HGT but the type carries NO
             seq/skip/sem information. Separates "more parameters" from "relation semantics".

Neither variant needs `edge_attr`, so both train on the plain `web_graphs.pkl`
(x / edge_index identical to the edge_attr pkl -- checked by scripts/build_edge_attr_graphs.py).
"""
import torch.nn as nn
from torch_geometric.nn import HGTConv

from src.bag.graph_builder import RELATION_TYPES
from src.models.baselines_graph import NODE_TYPE, HGTBaseline, _ConvBaseline
from src.models.layers import DEFAULT_HIDDEN, DEFAULT_IN, DEFAULT_OUT

MODES = ('collapsed', 'random')


def _hash_type(edge_index, n_types):
    """Deterministic pseudo-random edge type in [0, n_types): uninformative about seq/skip/sem."""
    src, dst = edge_index[0], edge_index[1]
    return ((src * 73856093) ^ (dst * 19349663)) % n_types


class HGTControl(HGTBaseline):
    needs_edge_attr = False

    def __init__(self, mode, in_channels=DEFAULT_IN, hidden_dim=DEFAULT_HIDDEN, out_channels=DEFAULT_OUT,
                 heads=8):
        assert mode in MODES, mode
        names = ('all',) if mode == 'collapsed' else RELATION_TYPES
        edge_types = [(NODE_TYPE, r, NODE_TYPE) for r in names]
        metadata = ([NODE_TYPE], edge_types)
        _ConvBaseline.__init__(self, HGTConv(in_channels, hidden_dim, metadata, heads=heads),
                               HGTConv(hidden_dim, hidden_dim, metadata, heads=heads),
                               hidden_dim, out_channels)
        self.edge_types = edge_types
        self.mode = mode

    def _convs(self, x, edge_index, edge_attr, conv):
        if self.mode == 'collapsed':
            edge_index_dict = {self.edge_types[0]: edge_index}
        else:
            rel = _hash_type(edge_index, len(self.edge_types))
            edge_index_dict = {et: edge_index[:, rel == i] for i, et in enumerate(self.edge_types)}
        return conv({NODE_TYPE: x}, edge_index_dict)[NODE_TYPE]


CONTROL_NAMES = {f'hgt_{m}': m for m in MODES}
