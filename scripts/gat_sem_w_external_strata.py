"""Where does gat_semW's external cost sit? Stratify the external set by whether the graph contains an E_sem edge
(= whether the W_sem path is ever active on it) and compare stock GATv2 / gat_placebo / gat_semW, 5 seeds each.

If the cost is a specialization trade-off it should concentrate where the added path is active (SQLi/XSS graphs with an
E_sem edge); a generic effect of leaving the stock recipe should also show on graphs where the path is never active
(Benign, SQLi/XSS without E_sem). Uncorrected Welch tests, n=5 vs 5.

Writes results/gat_sem_w_external_strata.txt.
"""
import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_ind

from scripts.eval_gat_sem_w import DEVICE, load
from scripts.hgt_investigation_common import EXTERNAL_EA, ROOT, SEEDS, load_pkl_graphs, logits_of

MODELS = ['gatv2', 'gat_placebo', 'gat_semW']


def main():
    g = load_pkl_graphs(EXTERNAL_EA)
    y = np.array([int(x.y) for x in g])
    has_sem = np.array([bool((x.edge_attr.argmax(1) == 2).any()) for x in g])
    strata = {
        'Benign (no E_sem anywhere)': (y == 0),
        'SQLi with E_sem': (y == 1) & has_sem,
        'SQLi without E_sem': (y == 1) & ~has_sem,
        'XSS with E_sem': (y == 2) & has_sem,
        'XSS without E_sem': (y == 2) & ~has_sem,
    }
    rec = {m: {k: [] for k in strata} for m in MODELS}
    for m in MODELS:
        for s in SEEDS:
            model, ea = load(m, s)
            p = logits_of(model, g, ea, DEVICE, batch_size=512).argmax(1).numpy()
            cls = {'Benign': 0, 'SQLi': 1, 'XSS': 2}
            for k, mask in strata.items():
                rec[m][k].append(float((p[mask] == cls[k.split()[0]]).mean()))
    L = ["External recall by stratum (mean ± std over 5 seeds); n per stratum: "
         + ", ".join(f"{k}={int(v.sum())}" for k, v in strata.items()), ""]
    rows = []
    for k in strata:
        a, b, c = (np.array(rec[m][k]) for m in MODELS)
        rows.append({'stratum': k, 'stock GATv2': f"{a.mean():.3f}±{a.std(ddof=1):.3f}",
                     'placebo': f"{b.mean():.3f}±{b.std(ddof=1):.3f}", 'semW': f"{c.mean():.3f}±{c.std(ddof=1):.3f}",
                     'semW-stock': f"{c.mean() - a.mean():+.3f}", 'p(semW vs stock)': f"{ttest_ind(c, a, equal_var=False).pvalue:.3f}",
                     'semW-placebo': f"{c.mean() - b.mean():+.3f}", 'p(semW vs placebo)': f"{ttest_ind(c, b, equal_var=False).pvalue:.3f}",
                     'placebo-stock': f"{b.mean() - a.mean():+.3f}", 'p(placebo vs stock)': f"{ttest_ind(b, a, equal_var=False).pvalue:.3f}"})
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        L.append(pd.DataFrame(rows).to_string(index=False))
    L += ["", "per-seed recall, SQLi with E_sem:"] + [f"  {m:12s} " + str([round(v, 3) for v in rec[m]['SQLi with E_sem']]) for m in MODELS]
    L += ["per-seed recall, SQLi without E_sem:"] + [f"  {m:12s} " + str([round(v, 3) for v in rec[m]['SQLi without E_sem']]) for m in MODELS]
    L += ["per-seed recall, Benign:"] + [f"  {m:12s} " + str([round(v, 3) for v in rec[m]['Benign (no E_sem anywhere)']]) for m in MODELS]
    print("\n".join(L))
    (ROOT / 'results' / 'gat_sem_w_external_strata.txt').write_text("\n".join(L) + "\n")


if __name__ == '__main__':
    main()
