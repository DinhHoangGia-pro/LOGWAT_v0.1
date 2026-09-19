"""Degree-corrected re-reading of the GATv2 layer-1 attention on the E_sem edges of the comment_split probe.

GATv2Conv softmax-normalizes attention over each destination node's incoming edges (self-loop included), so uniform
attention gives alpha = 1/in-degree. Comparing raw alpha across edge groups whose destination nodes differ in in-degree
(as the earlier 0.7614 / 0.8241 did) therefore measures degree, not preference. Reported per checkpoint:

  raw            alpha_E_sem_mean / alpha_other_mean, exactly as scripts/inspect_attention_weights.py and
                 run_ablation_edge_attr_seq_skip_sem.py computed it (self-loops excluded from 'other')
  raw if uniform the same ratio if attention were exactly uniform (mean 1/in-degree over the same edges)
  relative       alpha * in-degree(dst), mean over the group (1.0 = uniform), for E_sem / other / self-loops
  mass share     share of a destination node's attention mass carried by its E_sem in-edges, observed vs uniform
  per head       relative alpha on E_sem for each of the 8 heads (rules out opposing heads cancelling in the mean)

Checkpoints: the two that produced the old numbers (ablation_1 = retrain #5 without edge_attr, ablation_2 = with
edge_attr) and the five current seeds. Writes data/attention_degree_check.txt.
"""
import numpy as np
import torch

from scripts.train_baseline import ckpt_path
from src.bag.graph_builder import build_single_graph
from src.models.logwat import HeavyWebGNN

ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]
CONTENT = 'p=0;UNI/**/ON ALL SEL/**/ECT * FR/**/OM users WH/**/ERE id=0;--'
SEM = {(9, 16), (16, 9), (16, 23), (23, 16)}
MD = ROOT / 'data' / 'models_pretrained'


def measure(path, use_ea):
    model = HeavyWebGNN(use_edge_attr=use_ea)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    g = build_single_graph(CONTENT, use_edge_attr=use_ea)
    with torch.no_grad():
        if use_ea:
            _, (ei, al) = model.backbone.conv1(g.x, g.edge_index, g.edge_attr, return_attention_weights=True)
        else:
            _, (ei, al) = model.backbone.conv1(g.x, g.edge_index, return_attention_weights=True)
    al = al.numpy()
    a = al.mean(1)
    src, dst = ei[0].numpy(), ei[1].numpy()
    d = np.bincount(dst, minlength=g.num_nodes).astype(float)          # in-degree incl. self-loop
    sem = np.array([(int(s), int(t)) in SEM for s, t in zip(src, dst)])
    loop = src == dst
    oth = ~sem & ~loop
    rel = a * d[dst]
    obs, uni = [], []
    for v in sorted({int(t) for t in dst[sem]}):
        m = dst == v
        obs.append(a[m & sem].sum())
        uni.append((m & sem).sum() / m.sum())
    return dict(raw=a[sem].mean() / a[oth].mean(),
                null=(1 / d[dst][sem]).mean() / (1 / d[dst][oth]).mean(),
                rel_sem=rel[sem].mean(), rel_oth=rel[oth].mean(), rel_loop=rel[loop].mean(),
                indeg_sem=d[dst][sem].mean(), indeg_oth=d[dst][oth].mean(),
                mass_obs=float(np.mean(obs)), mass_uni=float(np.mean(uni)),
                per_head=(al * d[dst][:, None])[sem].mean(0))


def main():
    rows = [('config 1 = ablation_1 ckpt, no edge_attr (source of 0.7614)', MD / 'best_web_gnn_seed42_ablation_1_full_no_edge_attr.pth', False),
            ('config 2 = ablation_2 ckpt, +edge_attr (source of 0.8241)', MD / 'best_web_gnn_seed42_ablation_2_full_edge_attr.pth', True)]
    rows += [(f'current seed {s} (no edge_attr)', ckpt_path('gatv2', s), False) for s in (42, 43, 44, 45, 46)]
    L = [f"payload: {CONTENT!r}  (4 E_sem edges: {sorted(SEM)})", ""]
    for name, path, ea in rows:
        r = measure(path, ea)
        L.append(f"{name}")
        L.append(f"  raw ratio {r['raw']:.4f} | raw ratio if attention were uniform {r['null']:.4f} | mean in-degree of destination: E_sem {r['indeg_sem']:.2f} vs other {r['indeg_oth']:.2f}")
        L.append(f"  relative alpha (1.0 = uniform): E_sem {r['rel_sem']:.3f}, other {r['rel_oth']:.3f}, self-loop {r['rel_loop']:.3f}")
        L.append(f"  E_sem share of attention mass at its destination nodes: observed {r['mass_obs']:.3f} vs uniform {r['mass_uni']:.3f}")
        L.append(f"  per-head relative alpha on E_sem: {np.round(r['per_head'], 3).tolist()}  (min {r['per_head'].min():.3f}, max {r['per_head'].max():.3f})")
    report = "\n".join(L)
    print(report)
    (ROOT / 'data' / 'attention_degree_check.txt').write_text(report + "\n")


if __name__ == '__main__':
    main()
