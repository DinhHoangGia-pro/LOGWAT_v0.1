"""Extract real GATv2 attention weights (conv1, first layer) for the SQLi
comment_splitting sample, using the current checkpoint (retrain #5,
class-weighted loss). Compares attention on the 4 E_sem edges against
attention on all other edges, to see whether the model actually attends
more strongly to the semantic-keyword edges or treats them like any
other n-gram edge.
"""
from pathlib import Path

import torch

from scripts.investigate_causal_padding import build_graph, MODEL_PATH
from src.models.logwat import HeavyWebGNN

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
REPORT_PATH = DATA_DIR / 'attention_weights_comment_split.txt'

CONTENT = 'p=0;UNI/**/ON ALL SEL/**/ECT * FR/**/OM users WH/**/ERE id=0;--'
E_SEM_EDGES = {(9, 16), (16, 9), (16, 23), (23, 16)}


def main():
    lines = []
    lines.append(f"checkpoint: {MODEL_PATH}")
    lines.append(f"content: {CONTENT!r}")

    model = HeavyWebGNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    g = build_graph(CONTENT, 'attention_probe_comment_split')
    x, edge_index = g.x, g.edge_index
    lines.append(f"num_nodes={g.num_nodes} num_edges(original, no self-loops)={g.num_edges}")

    with torch.no_grad():
        out, (edge_index_att, alpha) = model.backbone.conv1(
            x, edge_index, return_attention_weights=True
        )

    lines.append(f"edge_index_att shape (after GATv2Conv self-loops): {tuple(edge_index_att.shape)}")
    lines.append(f"alpha shape [E, heads]: {tuple(alpha.shape)}")
    n_heads = alpha.shape[1]

    # per-edge attention, averaged across heads
    alpha_mean_per_edge = alpha.mean(dim=1)  # [E]

    src = edge_index_att[0].tolist()
    dst = edge_index_att[1].tolist()

    esem_alphas = []
    self_loop_alphas = []
    other_alphas = []

    for e in range(len(src)):
        i, j = src[e], dst[e]
        a = alpha_mean_per_edge[e].item()
        if (i, j) in E_SEM_EDGES:
            esem_alphas.append((i, j, a))
        elif i == j:
            self_loop_alphas.append((i, j, a))
        else:
            other_alphas.append((i, j, a))

    lines.append("")
    lines.append("=== E_sem edges: per-head + mean attention ===")
    for e in range(len(src)):
        i, j = src[e], dst[e]
        if (i, j) in E_SEM_EDGES:
            per_head = alpha[e].tolist()
            lines.append(f"  edge ({i},{j}): alpha_per_head={['%.4f' % v for v in per_head]} mean={alpha_mean_per_edge[e].item():.6f}")

    found_esem = {(i, j) for i, j, _ in esem_alphas}
    missing_esem = E_SEM_EDGES - found_esem
    if missing_esem:
        lines.append(f"  [!] E_sem edges NOT found in edge_index_att: {missing_esem}")

    esem_mean = sum(a for _, _, a in esem_alphas) / len(esem_alphas) if esem_alphas else float('nan')
    other_mean = sum(a for _, _, a in other_alphas) / len(other_alphas) if other_alphas else float('nan')
    self_loop_mean = sum(a for _, _, a in self_loop_alphas) / len(self_loop_alphas) if self_loop_alphas else float('nan')

    lines.append("")
    lines.append(f"n_esem_edges_found={len(esem_alphas)}, n_other_real_edges={len(other_alphas)}, n_self_loops={len(self_loop_alphas)}")
    lines.append(f"alpha_Esem_mean      = {esem_mean:.6f}")
    lines.append(f"alpha_other_mean     = {other_mean:.6f}  (excludes self-loops)")
    lines.append(f"alpha_self_loop_mean = {self_loop_mean:.6f}")
    lines.append(f"ratio alpha_Esem_mean / alpha_other_mean = {esem_mean / other_mean:.6f}")

    report = "\n".join(lines)
    print(report)
    REPORT_PATH.write_text(report + "\n")
    print(f"\nDa ghi: {REPORT_PATH}")


if __name__ == '__main__':
    main()
