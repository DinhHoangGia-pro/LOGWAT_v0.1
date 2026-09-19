"""Numbers behind the rewritten Fig. 6 case study, for the ORIGINAL case-study payload and the
comment-splitting payload used throughout the HGT investigation.

  P_intact  'UNION/**/SELECT/**/schema'   keywords intact, /**/ noise between them (original Fig. 6 payload)
  P_split   'p=0;UNI/**/ON ALL SEL/**/ECT * FR/**/OM users WH/**/ERE id=0;--'   keywords fragmented (held-out cell)

For each payload, 5 seeds: what the builder produces (edges by type), predictions / SQLi margin of GATv2 (LOGWAT),
typed HGT and the two same-architecture untyped controls, GATv2's first-layer attention on E_sem vs the other edges
(raw and degree-corrected, self-loops included as GATv2Conv adds them), and edge edits on the SAME checkpoints
(delete / re-type E_sem for HGT; delete E_sem / delete seq for GATv2).

Writes data/case_study_fig6.txt.
"""
import numpy as np
import torch

from scripts.hgt_investigation_common import (DATA_DIR, SEEDS, apply_condition, load_model, logits_of, margin,
                                             uses_edge_attr)
from src.bag.graph_builder import build_single_graph
from src.preprocessing.tokenizer import web_security_tokenizer

PAYLOADS = {
    'P_intact': 'UNION/**/SELECT/**/schema',
    'P_split': 'p=0;UNI/**/ON ALL SEL/**/ECT * FR/**/OM users WH/**/ERE id=0;--',
}
NAMES = ['Benign', 'SQLi', 'XSS']


def gat_attention(model, g):
    """GATv2Conv layer-1 head-mean attention; -> (raw ratio E_sem/other, relative alpha*indeg on E_sem, relative on others)."""
    with torch.no_grad():
        _, (ei, alpha) = model.backbone.conv1(g.x, g.edge_index, return_attention_weights=True)
    a = alpha.mean(1)
    src, dst = ei[0], ei[1]
    sem_pairs = {(int(s), int(d)) for s, d, r in zip(g.edge_index[0], g.edge_index[1], g.edge_attr.argmax(1)) if r == 2}
    is_sem = torch.tensor([(int(s), int(d)) in sem_pairs for s, d in zip(src, dst)])
    other = (~is_sem) & (src != dst)                       # real non-sem edges, no self-loops (as in the earlier probe)
    indeg = torch.bincount(dst, minlength=g.num_nodes).float()   # includes the self-loop
    rel = a * indeg[dst]
    return (a[is_sem].mean() / a[other].mean()).item(), rel[is_sem].mean().item(), rel[other].mean().item()


def main():
    L = []
    out = lambda s='': (L.append(s), print(s))
    for pname, text in PAYLOADS.items():
        g = build_single_graph(text, source_uid=pname, attack_type=1, use_edge_attr=True)
        rel = g.edge_attr.argmax(1)
        counts = {n: int((rel == i).sum()) for i, n in enumerate(('seq', 'skip', 'sem'))}
        sem = sorted((int(s), int(d)) for s, d, r in zip(g.edge_index[0], g.edge_index[1], rel) if r == 2)
        toks = web_security_tokenizer(text)
        out(f"=== {pname}: {text!r}")
        out(f"tokens ({len(toks)}): " + " ".join(f"{i}:{t}" for i, t in enumerate(toks)))
        out(f"edges {counts} (total {g.num_edges}); E_sem share = {counts['sem'] / g.num_edges:.3f}; "
            f"non-sem : sem = {(g.num_edges - counts['sem']) / max(counts['sem'], 1):.0f}:1; E_sem edges = {sem}")
        for key in ('gatv2', 'hgt', 'hgt_collapsed', 'hgt_random'):
            preds, margins = [], []
            for s in SEEDS:
                lg = logits_of(load_model(key, s), [g], uses_edge_attr(key))
                preds.append(NAMES[int(lg.argmax())][0])
                margins.append(margin(lg, 1).item())
            out(f"  {key:14s} pred by seed 42..46 = {''.join(preds)}   SQLi margin per seed = {[round(m, 2) for m in margins]}  mean {np.mean(margins):+.2f}")
        # GATv2 attention, 5 seeds
        att = [gat_attention(load_model('gatv2', s), g) for s in SEEDS]
        out(f"  GATv2 conv1 attention on E_sem: raw ratio sem/other per seed = {[round(a[0], 3) for a in att]}; "
            f"degree-corrected (alpha*indeg, 1.0 = uniform incl. self-loop) sem = {np.mean([a[1] for a in att]):.3f}, other = {np.mean([a[2] for a in att]):.3f}")
        # edge edits
        for key, conds in (('hgt', ('real', 'drop_sem', 'sem_as_seq', 'sem_as_skip')), ('gatv2', ('real', 'drop_sem', 'drop_seq'))):
            for cond in conds:
                gg = apply_condition(g, cond)
                preds, margins = [], []
                for s in SEEDS:
                    lg = logits_of(load_model(key, s), [gg], uses_edge_attr(key))
                    preds.append(NAMES[int(lg.argmax())][0])
                    margins.append(margin(lg, 1).item())
                out(f"  edit {key:5s} {cond:11s} pred = {''.join(preds)}  mean SQLi margin {np.mean(margins):+.2f}")
        out()
    # ---- the other #16 baselines on the same two payloads (the original case study claimed Bi-LSTM / GCN failures) ----
    from scripts.evaluate_baselines import load_model as load_baseline, needs_edge_attr, predict, texts_to_data
    out("=== other #16 baselines on the same payloads (pred by seed 42..46; S=SQLi B=Benign X=XSS) ===")
    for pname, text in PAYLOADS.items():
        row = []
        for label, key in (('GCN', 'gcn'), ('GraphSAGE', 'graphsage'), ('GIN', 'gin'), ('TextCNN', 'textcnn'),
                           ('Bi-LSTM', 'bilstm'), ('StackLSTM', 'stacklstm')):
            data = texts_to_data(key, [text], [pname], [1])
            preds = ''.join(NAMES[predict(load_baseline(key, s, 'cpu'), data, needs_edge_attr(key), 'cpu')[0]][0] for s in SEEDS)
            row.append(f"{label}={preds}")
        out(f"  {pname}: " + "  ".join(row))
    (DATA_DIR / 'case_study_fig6.txt').write_text("\n".join(L) + "\n")
    print("[+] wrote data/case_study_fig6.txt")


if __name__ == '__main__':
    main()
