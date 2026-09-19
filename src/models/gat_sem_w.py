"""Minimal causal probe: GATv2 with a SEPARATE message matrix for flagged edges, attention untouched.

`GATv2ConvAltMsg` is `GATv2Conv` with exactly one change: on flagged edges the message value is
`W_alt x_j` instead of `W_l x_j`. The attention logits, softmax, self-loop handling, bias and everything
else are the stock computation (alpha is still computed from `lin_l` / `lin_r`), so the architecture cannot
change HOW MUCH an edge is attended to -- only WHAT it carries. With `lin_alt` tied to `lin_l` (or no flagged
edge) the layer is numerically identical to `GATv2Conv` (checked in scripts/train_gat_sem_w.py --selftest).

Arms (which edges are flagged):
  sem      E_sem edges, read from the relation one-hot `edge_attr` (argmax == 2). `edge_attr` is used ONLY to build
           this mask; it never enters attention (the convs have no edge_dim).
  placebo  a hash of the (src, dst) node index within each graph selects ~0.405% of edges (the measured E_sem share of all training
           edges, 14,260 / 3,520,497) -- same extra parameters, same flagged-edge rate, no relation information.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GATv2Conv, JumpingKnowledge
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import add_self_loops, remove_self_loops

from src.models.layers import Classifier, DEFAULT_HEADS, DEFAULT_HIDDEN, DEFAULT_IN, DEFAULT_OUT

PLACEBO_MOD = 247      # 1 / 0.00405 = 246.9 -> ~0.405% of edges flagged


class GATv2ConvAltMsg(GATv2Conv):
    def __init__(self, in_channels, out_channels, heads):
        super().__init__(in_channels, out_channels, heads=heads)    # same as GATBackbone's conv: no edge_dim
        self.lin_alt = Linear(in_channels, heads * out_channels, bias=True, weight_initializer='glorot')
        self.lin_alt.reset_parameters()

    def forward(self, x, edge_index, alt_mask, return_attention_weights=False):
        H, C = self.heads, self.out_channels
        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)
        x_alt = self.lin_alt(x).view(-1, H, C)
        edge_index, alt_mask = remove_self_loops(edge_index, alt_mask)
        edge_index, alt_mask = add_self_loops(edge_index, alt_mask, fill_value=0.0, num_nodes=x.size(0))
        # edge_updater_type: (x: PairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, x=(x_l, x_r), edge_attr=None)
        # propagate_type: (x: PairTensor, x_alt: Tensor, alt: Tensor, alpha: Tensor)
        out = self.propagate(edge_index, x=(x_l, x_r), x_alt=x_alt, alt=alt_mask.view(-1, 1, 1).to(x_l.dtype), alpha=alpha)
        out = out.view(-1, H * C)
        if self.bias is not None:
            out = out + self.bias
        return (out, (edge_index, alpha)) if return_attention_weights else out

    def message(self, x_j, x_alt_j, alt, alpha):
        return (alt * x_alt_j + (1.0 - alt) * x_j) * alpha.unsqueeze(-1)


class AltMsgBackbone(nn.Module):
    def __init__(self, in_channels=DEFAULT_IN, hidden_dim=DEFAULT_HIDDEN, heads=DEFAULT_HEADS):
        super().__init__()
        self.conv1 = GATv2ConvAltMsg(in_channels, hidden_dim // heads, heads)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GATv2ConvAltMsg(hidden_dim, hidden_dim // heads, heads)
        self.bn2 = BatchNorm(hidden_dim)
        self.jk = JumpingKnowledge(mode='cat')

    def forward(self, x, edge_index, alt_mask):
        x1 = F.elu(self.bn1(self.conv1(x, edge_index, alt_mask)))
        x2 = F.elu(self.bn2(self.conv2(x1, edge_index, alt_mask)))
        return self.jk([x1, x2])


class GATAltMsgNet(nn.Module):
    """LOGWAT's HeavyWebGNN with `GATv2ConvAltMsg` layers; arm in {'sem', 'placebo'}."""
    needs_edge_attr = True      # only to derive the E_sem mask

    def __init__(self, arm, in_channels=DEFAULT_IN, hidden_dim=DEFAULT_HIDDEN, out_channels=DEFAULT_OUT):
        super().__init__()
        assert arm in ('sem', 'placebo'), arm
        self.arm = arm
        self.backbone = AltMsgBackbone(in_channels, hidden_dim)
        self.classifier = Classifier(hidden_dim=hidden_dim, out_channels=out_channels)

    def alt_mask(self, edge_index, edge_attr, batch):
        if self.arm == 'sem':
            if edge_attr is None:
                raise ValueError("arm 'sem' needs edge_attr (relation one-hot) to locate E_sem edges")
            return (edge_attr.argmax(dim=1) == 2).float()
        # placebo: hash of the node's index WITHIN its own graph, so the flagged set is a fixed function of the
        # graph, independent of how graphs are batched
        counts = torch.bincount(batch, minlength=int(batch.max()) + 1)
        ptr = torch.cumsum(counts, 0) - counts
        local = torch.arange(batch.numel(), device=batch.device) - ptr[batch]
        s, d = local[edge_index[0]], local[edge_index[1]]
        return ((((s * 73856093) ^ (d * 19349663)) % PLACEBO_MOD) == 0).float()

    def forward(self, x, edge_index, batch, edge_attr=None):
        feats = self.backbone(x, edge_index, self.alt_mask(edge_index, edge_attr, batch))
        return self.classifier(feats, batch)


ARMS = {'gat_semW': 'sem', 'gat_placebo': 'placebo'}
