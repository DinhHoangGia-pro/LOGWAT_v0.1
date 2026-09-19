"""Evaluate the HGT control variants next to typed HGT and GATv2 (5 seeds), on the same three modes
as the #16 baselines (frozen test split / held-out 90-row matrix / external dataset).

  hgt            typed HGT (real seq/skip/sem edge types)          -- the #16 baseline
  hgt_collapsed  same HGT, ONE edge type                            -- relation information removed
  hgt_random     same HGT, 3 edge types assigned by a node-index hash -- same #params, no relation information
  gatv2          deployed LOGWAT GATv2

If typed HGT's held-out result comes from its edge TYPES, the two controls (same attention form, same
K/Q/V, same skip gate, same training loop) should lose it; if it comes from HGT's architecture in general,
they should keep it.

Writes results/hgt_controls_5seed.csv (one row per model x seed) and data/hgt_controls_report.txt.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from scripts.hgt_investigation_common import (EXTERNAL_EA, ROOT, SEEDS, apply_condition, cell_correct, load_model,
                                             load_pkl_graphs, logits_of, margin, matrix_graphs, test_split_graphs,
                                             uses_edge_attr)
from scripts.investigate_hgt_generalization import available

MODELS = ['gatv2', 'hgt', 'hgt_collapsed', 'hgt_random']
CS, CM = 'SQLi/comment_splitting', 'SQLi/case_mixing'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    df_m, m_graphs = matrix_graphs()
    t_graphs = test_split_graphs()
    y_t = np.array([int(g.y) for g in t_graphs])
    e_graphs = load_pkl_graphs(EXTERNAL_EA)
    y_e = np.array([int(g.y) for g in e_graphs])
    cs_mask = torch.tensor(((df_m['class_name'] == 'SQLi') & (df_m['technique'] == 'comment_splitting')).values)

    rows = []
    for key in MODELS:
        for seed in SEEDS:
            if not available(key, seed):
                print(f"[!] {key} seed {seed}: no checkpoint yet, skipping", flush=True)
                continue
            model = load_model(key, seed, DEVICE)
            ea = uses_edge_attr(key)
            lg_m = logits_of(model, m_graphs, ea, DEVICE)
            cells = cell_correct(df_m, lg_m.argmax(1).numpy())
            test_acc = float((logits_of(model, t_graphs, ea, DEVICE).argmax(1).numpy() == y_t).mean())
            pe = logits_of(model, e_graphs, ea, DEVICE, batch_size=512).argmax(1).numpy()
            row = dict(model=key, seed=seed, test_acc=test_acc, heldout_correct=sum(cells.values()),
                       heldout_full_cells=sum(1 for v in cells.values() if v == 10),
                       cs_correct=cells[CS], cm_correct=cells[CM], cs_margin=margin(lg_m[cs_mask], 1).mean().item(),
                       ext_wF1=f1_score(y_e, pe, average='weighted'), ext_benign_recall=float((pe[y_e == 0] == 0).mean()),
                       ext_sqli_recall=float((pe[y_e == 1] == 1).mean()), ext_xss_recall=float((pe[y_e == 2] == 2).mean()))
            row.update({f"cell:{k}": v for k, v in cells.items()})
            rows.append(row)
            print(f"[{key} seed {seed}] test={test_acc:.4f} heldout={row['heldout_correct']}/90 cs={row['cs_correct']}/10 "
                  f"cm={row['cm_correct']}/10 cs_margin={row['cs_margin']:+.2f} extwF1={row['ext_wF1']:.3f} "
                  f"benign_rec={row['ext_benign_recall']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / 'results' / 'hgt_controls_5seed.csv', index=False)
    num = ['test_acc', 'heldout_correct', 'heldout_full_cells', 'cs_correct', 'cm_correct', 'cs_margin', 'ext_wF1',
           'ext_benign_recall', 'ext_sqli_recall', 'ext_xss_recall']
    agg = out.groupby('model', sort=False)[num].agg(['mean', 'std'])
    agg.insert(0, 'n_seeds', out.groupby('model', sort=False).size())
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        lines = ["5-seed mean/std per model (cs/cm = correct out of 10 rows of that cell, per seed):",
                 agg.round(3).to_string(), "",
                 "per-seed:", out[['model', 'seed'] + num].round(3).to_string(index=False), "",
                 "per-cell correct out of 10, summed over seeds (/10 x n_seeds):"]
        cellcols = [c for c in out.columns if c.startswith('cell:')]
        lines.append(out.groupby('model', sort=False)[cellcols].sum().rename(columns=lambda c: c[5:]).T.to_string())
    report = "\n".join(lines)
    print("\n" + report)
    (ROOT / 'data' / 'hgt_controls_report.txt').write_text(report + "\n")


if __name__ == '__main__':
    main()
