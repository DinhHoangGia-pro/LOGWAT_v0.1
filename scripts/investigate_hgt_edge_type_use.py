"""Causal test: does the TRAINED typed-HGT actually use its edge types, and which type?

Same checkpoints (data/models_pretrained/baseline_hgt_seed{42..46}.pth), same graphs, only the
relation label / presence of edges is edited at inference:

  real            unchanged (must reproduce the published 90/90 -- sanity check)
  all_seq/skip/sem  every edge relabelled to ONE type  (removes all relation information)
  cyclic          seq->skip, skip->sem, sem->seq       (all information kept, but mis-assigned)
  swap_seq_skip   seq<->skip, sem untouched
  sem_as_seq/skip only the 4-ish E_sem edges are relabelled
  random          hash-assigned types (chance agreement with the truth)
  drop_sem/skip/seq  delete that edge type

If HGT solves SQLi/comment_splitting BECAUSE it separates relation types, then relabelling / dropping
must hurt it. GATv2 (relation-agnostic) gets the drop_* conditions for contrast.

Writes results/hgt_edge_type_intervention.csv (long) and data/hgt_edge_type_intervention.txt.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.hgt_investigation_common import (CLASS_NAMES, SEEDS, ROOT, apply_condition, cell_correct, load_model,
                                             logits_of, margin, matrix_graphs, test_split_graphs, uses_edge_attr)

CONDITIONS = ['real', 'all_seq', 'all_skip', 'all_sem', 'cyclic', 'swap_seq_skip', 'sem_as_seq', 'sem_as_skip',
              'random', 'drop_sem', 'drop_skip', 'drop_seq']
GATV2_CONDITIONS = ['real', 'drop_sem', 'drop_skip', 'drop_seq']
CS = 'SQLi/comment_splitting'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-test-split', action='store_true')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df, m_graphs = matrix_graphs()
    y_m = df['attack_type'].values
    t_graphs = None if args.skip_test_split else test_split_graphs()
    y_t = None if t_graphs is None else np.array([int(g.y) for g in t_graphs])

    rows = []
    for key, conds in (('hgt', CONDITIONS), ('gatv2', GATV2_CONDITIONS)):
        for seed in SEEDS:
            model = load_model(key, seed, device)
            use_ea = uses_edge_attr(key)
            for cond in conds:
                t0 = time.time()
                lg = logits_of(model, [apply_condition(g, cond) for g in m_graphs], use_ea, device)
                pred = lg.argmax(1).numpy()
                cells = cell_correct(df, pred)
                cs_mask = ((df['class_name'] == 'SQLi') & (df['technique'] == 'comment_splitting')).values
                cs_margin = margin(lg[torch.tensor(cs_mask)], 1).mean().item()
                for cell, n in cells.items():
                    rows.append(dict(model=key, seed=seed, cond=cond, dataset='matrix', cell=cell, n_correct=n, n_total=10,
                                     cs_margin=cs_margin if cell == CS else np.nan))
                test_acc = np.nan
                if t_graphs is not None:
                    pt = logits_of(model, [apply_condition(g, cond) for g in t_graphs], use_ea, device).argmax(1).numpy()
                    test_acc = float((pt == y_t).mean())
                    rows.append(dict(model=key, seed=seed, cond=cond, dataset='test_split', cell='all',
                                     n_correct=int((pt == y_t).sum()), n_total=len(y_t), cs_margin=np.nan))
                print(f"[{key} seed {seed}] {cond:14s} matrix={sum(cells.values())}/90  {CS}={cells[CS]}/10 "
                      f"margin_SQLi={cs_margin:+.2f}  test_acc={test_acc:.4f}  ({time.time() - t0:.1f}s)", flush=True)

    out = pd.DataFrame(rows)
    csv = ROOT / 'results' / 'hgt_edge_type_intervention.csv'
    out.to_csv(csv, index=False)

    lines = ["Causal edge-type interventions on TRAINED checkpoints (5 seeds, matrix = 10 rows/cell/seed -> /50).",
             "cs_margin = mean over the 10 comment_splitting rows of logit[SQLi] - max(other logits); >0 means SQLi wins.", ""]
    for key in ('hgt', 'gatv2'):
        sub = out[(out.model == key) & (out.dataset == 'matrix')]
        piv = sub.pivot_table(index='cond', columns='cell', values='n_correct', aggfunc='sum', sort=False)
        piv['total/450'] = piv.sum(axis=1)
        marg = sub[sub.cell == CS].groupby('cond', sort=False)['cs_margin'].agg(['mean', 'std'])
        piv['cs_margin_mean'] = marg['mean']
        piv['cs_margin_sd'] = marg['std']
        tst = out[(out.model == key) & (out.dataset == 'test_split')].groupby('cond', sort=False)['n_correct'].sum()
        if len(tst):
            piv['test_acc'] = tst / (len(y_t) * len(SEEDS))
        lines += [f"--- {key} ---", piv.round(3).to_string(), ""]
    report = "\n".join(lines)
    print("\n" + report)
    (ROOT / 'data' / 'hgt_edge_type_intervention.txt').write_text(report + "\n")
    print(f"[+] wrote {csv} and data/hgt_edge_type_intervention.txt")


if __name__ == '__main__':
    main()
