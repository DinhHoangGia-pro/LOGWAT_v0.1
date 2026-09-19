"""Extract REAL HGT attention weights (both HGTConv layers, all heads) for the SQLi
comment_splitting probe used for GATv2 in scripts/inspect_attention_weights.py, and break
them down by edge TYPE (seq / skip / sem).

HGTConv has no `return_attention_weights`. To get the alpha it actually applies -- not a
re-implementation of it -- `torch_geometric.nn.conv.hgt_conv.softmax` (the softmax over each
destination node's incoming edges inside `HGTConv.message`) is wrapped for the duration of one
forward pass and its output recorded. The hooked forward is checked to be bit-identical to the
un-hooked one. Edge order inside HGTConv is the concatenation over edge types (dict order), so
the type of each recorded alpha row is known exactly.

Reports, per seed / layer / edge type:
  raw alpha (head-mean)         directly comparable to GATv2's `alpha_Esem_mean / alpha_other_mean`
                                (the GATv2 numbers include self-loops in the softmax denominator, HGT has none)
  relative alpha = alpha*indeg  1.0 == uniform attention over the destination's neighbours; removes the
                                1/in-degree effect that makes raw means of sparse edge types look small
  mass share                    fraction of all attention mass carried by that edge type vs its share of edges

Attention is NOT influence (the value transform W_msg is also per-edge-type), so this is only
descriptive; causal tests are in scripts/investigate_hgt_edge_type_use.py.
"""
from pathlib import Path

import torch
from torch_geometric.nn.conv import hgt_conv

from scripts.train_baseline import ckpt_path
from src.bag.graph_builder import RELATION_TYPES, build_single_graph
from src.models.baselines_graph import HGTBaseline
from src.preprocessing.tokenizer import web_security_tokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
REPORT_PATH = DATA_DIR / 'attention_weights_comment_split_hgt.txt'

CONTENT = 'p=0;UNI/**/ON ALL SEL/**/ECT * FR/**/OM users WH/**/ERE id=0;--'
E_SEM_EDGES = {(9, 16), (16, 9), (16, 23), (23, 16)}  # same 4 edges as the GATv2 probe
SEEDS = [42, 43, 44, 45, 46]


def load_hgt(seed):
    model = HGTBaseline()
    model.load_state_dict(torch.load(ckpt_path('hgt', seed), map_location='cpu'))
    return model.eval()


def forward_with_attention(model, g):
    """-> (logits, per_layer) ; per_layer[l] = dict(alpha=[E,H], src, dst, etype) in HGTConv edge order."""
    orig_softmax = hgt_conv.softmax
    rec = []

    def recording_softmax(src, index, ptr=None, num_nodes=None, dim=0):
        out = orig_softmax(src, index, ptr, num_nodes, dim)
        rec.append((out.detach().clone(), index.detach().clone()))
        return out

    rel = g.edge_attr.argmax(dim=1)
    order = torch.cat([torch.nonzero(rel == i).flatten() for i in range(len(RELATION_TYPES))])
    hgt_conv.softmax = recording_softmax
    try:
        with torch.no_grad():
            logits = model(g.x, g.edge_index, torch.zeros(g.num_nodes, dtype=torch.long), edge_attr=g.edge_attr)
    finally:
        hgt_conv.softmax = orig_softmax
    assert len(rec) == 2, f"expected 2 HGTConv layers -> 2 softmax calls, got {len(rec)}"
    layers = []
    for alpha, dst_idx in rec:
        layers.append(dict(alpha=alpha, dst=dst_idx, src=g.edge_index[0][order], etype=rel[order],
                           dst_check=g.edge_index[1][order]))
        assert torch.equal(dst_idx, layers[-1]['dst_check']), "edge order assumption violated"
    return logits, layers


def summarize(layer, n_nodes):
    alpha = layer['alpha'].mean(dim=1)              # head-mean, [E]
    dst, etype = layer['dst'], layer['etype']
    indeg = torch.bincount(dst, minlength=n_nodes).float()
    rel_alpha = alpha * indeg[dst]                   # 1.0 == uniform over the node's neighbours
    total_mass = alpha.sum().item()
    out = {}
    for i, name in enumerate(RELATION_TYPES):
        m = etype == i
        out[name] = dict(n=int(m.sum()),
                         raw_mean=alpha[m].mean().item() if m.any() else float('nan'),
                         rel_mean=rel_alpha[m].mean().item() if m.any() else float('nan'),
                         mass_share=(alpha[m].sum().item() / total_mass) if m.any() else float('nan'),
                         edge_share=m.float().mean().item())
    return out


def main():
    g = build_single_graph(CONTENT, source_uid='attention_probe_comment_split', attack_type=1, use_edge_attr=True)
    tokens = web_security_tokenizer(CONTENT)
    sem = {(int(a), int(b)) for a, b, r in zip(g.edge_index[0], g.edge_index[1], g.edge_attr.argmax(1)) if r == 2}
    lines = [f"content: {CONTENT!r}",
             f"num_nodes={g.num_nodes} num_edges={g.num_edges}  "
             f"(seq={(g.edge_attr.argmax(1)==0).sum().item()}, skip={(g.edge_attr.argmax(1)==1).sum().item()}, sem={len(sem)})",
             f"E_sem edges in the current builder: {sorted(sem)}  == GATv2-probe E_SEM_EDGES: {sem == E_SEM_EDGES}",
             "tokens: " + ", ".join(f"{i}:{t}" for i, t in enumerate(tokens)),
             "GATv2 reference (same probe): ratio alpha_Esem/alpha_other = 0.7614 (config 1, no edge_attr), 0.8241 (config 2, +edge_attr)",
             ""]
    agg = {l: {n: [] for n in RELATION_TYPES} for l in (0, 1)}
    for seed in SEEDS:
        model = load_hgt(seed)
        with torch.no_grad():
            plain = model(g.x, g.edge_index, torch.zeros(g.num_nodes, dtype=torch.long), edge_attr=g.edge_attr)
        logits, layers = forward_with_attention(model, g)
        assert torch.equal(plain, logits), "hook changed the forward result"
        pred = int(logits.argmax())
        lines.append(f"=== HGT seed {seed}: pred={['Benign','SQLi','XSS'][pred]} logits={[round(v, 3) for v in logits[0].tolist()]} "
                     f"(hooked forward == plain forward: True) ===")
        for li, layer in enumerate(layers):
            s = summarize(layer, g.num_nodes)
            other_raw = torch.cat([layer['alpha'].mean(1)[layer['etype'] == i] for i in (0, 1)]).mean().item()
            lines.append(f"  layer {li + 1}: ratio raw alpha_sem/alpha_(seq+skip) = {s['sem']['raw_mean'] / other_raw:.4f}")
            for name in RELATION_TYPES:
                d = s[name]
                lines.append(f"    {name:4s} n={d['n']:3d}  raw_mean={d['raw_mean']:.4f}  relative(alpha*indeg)={d['rel_mean']:.3f}  "
                             f"mass_share={d['mass_share']:.4f} vs edge_share={d['edge_share']:.4f}")
                agg[li][name].append(d)
            if li == 0:
                per_head = layer['alpha'][layer['etype'] == 2]   # [4, H]
                lines.append(f"    E_sem per-head alpha, layer 1 (rows = the 4 sem edges): "
                             + "; ".join("[" + ",".join(f"{v:.3f}" for v in row) + "]" for row in per_head.tolist()))
        lines.append("")

    lines.append("=== 5-seed mean of the per-seed numbers above ===")
    for li in (0, 1):
        lines.append(f"layer {li + 1}:")
        for name in RELATION_TYPES:
            rows = agg[li][name]
            mean = lambda k: sum(r[k] for r in rows) / len(rows)
            lines.append(f"  {name:4s} raw_mean={mean('raw_mean'):.4f}  relative={mean('rel_mean'):.3f}  "
                         f"mass_share={mean('mass_share'):.4f} (edge_share {mean('edge_share'):.4f})")
        other = (sum(r['raw_mean'] for n in ('seq', 'skip') for r in agg[li][n]) / (2 * len(SEEDS)))
        lines.append(f"  ratio raw alpha_sem / alpha_(seq+skip), mean of seeds = "
                     f"{sum(r['raw_mean'] for r in agg[li]['sem']) / len(SEEDS) / other:.4f}")
    report = "\n".join(lines)
    print(report)
    REPORT_PATH.write_text(report + "\n")
    print(f"\nDa ghi: {REPORT_PATH}")


if __name__ == '__main__':
    main()
