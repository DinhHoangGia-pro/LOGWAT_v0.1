"""Which edge-type CHANNEL decides typed-HGT's output on 1-2-token external Benign graphs?

Those graphs (75% of external Benign; none exist in the training set for Benign/SQLi) contain only
seq-typed edges (2 tokens: 2 directed seq edges; 1 token: the builder's typed-seq self-loop). Relabelling
those edges to skip / sem at inference shows the prediction is a function of the type channel the single
edge lands in, and that the seq-channel answer differs by seed.

Writes data/hgt_tiny_graph_channel_probe.txt.
"""
import numpy as np
import torch

from scripts.hgt_investigation_common import (DATA_DIR, EXTERNAL_EA, SEEDS, apply_condition, load_model,
                                             load_pkl_graphs, logits_of)

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    lines = ["External Benign graphs with 1 or 2 tokens (only seq-typed edges exist: n=2 -> 2 directed seq edges, "
             "n=1 -> builder's typed-seq self-loop).",
             "Typed HGT, predictions when those edges keep their real type (seq) vs are relabelled skip / sem. "
             "cols = % predicted Benign/SQLi/XSS"]
    g = load_pkl_graphs(EXTERNAL_EA)
    y = np.array([int(x.y) for x in g])
    n = np.array([x.num_nodes for x in g])
    for size in (1, 2):
        idx = np.where((y == 0) & (n == size))[0]
        gs = [g[i] for i in idx]
        lines.append(f"--- n={size} tokens ({len(idx)} benign graphs)")
        for seed in SEEDS:
            m = load_model('hgt', seed, DEV)
            row = []
            for cond in ('real', 'all_skip', 'all_sem'):
                p = logits_of(m, [apply_condition(x, cond) for x in gs], True, DEV).argmax(1).numpy()
                c = np.bincount(p, minlength=3) / len(p)
                row.append(f"{cond}: B/S/X={c[0] * 100:5.1f}/{c[1] * 100:5.1f}/{c[2] * 100:5.1f}")
            lines.append(f"  seed {seed}: " + " | ".join(row))
    report = "\n".join(lines)
    print(report)
    (DATA_DIR / 'hgt_tiny_graph_channel_probe.txt').write_text(report + "\n")


if __name__ == '__main__':
    main()
