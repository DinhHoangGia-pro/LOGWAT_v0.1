"""Evaluate the minimal-intervention GATv2 variants next to stock GATv2 (5 seeds each):

  gatv2        deployed LOGWAT GATv2 (shared message matrix for every edge)
  gat_semW     GATv2 whose MESSAGE on E_sem edges uses its own matrix W_sem; attention computation untouched
  gat_placebo  same extra parameters, but the separate matrix is applied to hash-selected edges (~0.4%) that carry
               no relation information

Question: does giving E_sem its own message matrix -- and nothing else -- repair SQLi/comment_splitting?
Reports per model x seed: test acc, held-out matrix per cell, comment_splitting pass + SQLi margin, external wF1 /
Benign recall, the two case-study payloads, GATv2-style layer-1 attention on E_sem (degree-corrected) AFTER training,
and edits on the trained gat_semW (route E_sem back through W_l / delete E_sem).

Writes results/gat_sem_w_5seed.csv, data/gat_sem_w_report.txt.
"""
import numpy as np
import pandas as pd
import torch
from scipy.stats import fisher_exact
from sklearn.metrics import f1_score

from scripts.hgt_investigation_common import (EXTERNAL_EA, ROOT, SEEDS, apply_condition, cell_correct, load_model,
                                             load_pkl_graphs, logits_of, margin, matrix_graphs, test_split_graphs)
from scripts.train_gat_sem_w import ckpt_path
from src.bag.graph_builder import build_single_graph
from src.models.gat_sem_w import ARMS, GATAltMsgNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS = ['gatv2', 'gat_semW', 'gat_placebo']
CS, CM = 'SQLi/comment_splitting', 'SQLi/case_mixing'
P_SPLIT = 'p=0;UNI/**/ON ALL SEL/**/ECT * FR/**/OM users WH/**/ERE id=0;--'
P_INTACT = 'UNION/**/SELECT/**/schema'


def load(key, seed):
    if key == 'gatv2':
        return load_model('gatv2', seed, DEVICE), False
    m = GATAltMsgNet(ARMS[key])
    m.load_state_dict(torch.load(ckpt_path(key, seed), map_location=DEVICE))
    return m.to(DEVICE).eval(), True


def sem_attention(model, g):
    """Layer-1 head-mean attention on the 4 E_sem edges: (raw ratio vs other edges, degree-corrected alpha*indeg)."""
    sem_pairs = {(int(s), int(d)) for s, d, r in zip(g.edge_index[0], g.edge_index[1], g.edge_attr.argmax(1)) if r == 2}
    with torch.no_grad():
        if isinstance(model, GATAltMsgNet):
            mask = model.alt_mask(g.edge_index, g.edge_attr, torch.zeros(g.num_nodes, dtype=torch.long))
            _, (ei, al) = model.backbone.conv1(g.x, g.edge_index, mask, return_attention_weights=True)
        else:
            _, (ei, al) = model.backbone.conv1(g.x, g.edge_index, return_attention_weights=True)
    a = al.mean(1).numpy()
    src, dst = ei[0].numpy(), ei[1].numpy()
    d = np.bincount(dst, minlength=g.num_nodes).astype(float)
    sem = np.array([(int(s), int(t)) in sem_pairs for s, t in zip(src, dst)])
    oth = ~sem & (src != dst)
    return a[sem].mean() / a[oth].mean(), (a * d[dst])[sem].mean()


def main():
    df_m, m_graphs = matrix_graphs()
    t_graphs = test_split_graphs()
    y_t = np.array([int(g.y) for g in t_graphs])
    e_graphs = load_pkl_graphs(EXTERNAL_EA)
    y_e = np.array([int(g.y) for g in e_graphs])
    cs_mask = torch.tensor(((df_m['class_name'] == 'SQLi') & (df_m['technique'] == 'comment_splitting')).values)
    probes = {n: build_single_graph(t, source_uid=n, attack_type=1, use_edge_attr=True)
              for n, t in (('P_split', P_SPLIT), ('P_intact', P_INTACT))}

    rows = []
    for key in MODELS:
        for seed in SEEDS:
            if key != 'gatv2' and not ckpt_path(key, seed).exists():
                print(f"[!] {key} seed {seed}: no checkpoint yet", flush=True)
                continue
            model, ea = load(key, seed)
            lg_m = logits_of(model, m_graphs, ea, DEVICE)
            cells = cell_correct(df_m, lg_m.argmax(1).numpy())
            pe = logits_of(model, e_graphs, ea, DEVICE, batch_size=512).argmax(1).numpy()
            row = dict(model=key, seed=seed,
                       test_acc=float((logits_of(model, t_graphs, ea, DEVICE).argmax(1).numpy() == y_t).mean()),
                       heldout_correct=sum(cells.values()), cs_correct=cells[CS], cm_correct=cells[CM],
                       cs_margin=margin(lg_m[cs_mask], 1).mean().item(),
                       ext_wF1=f1_score(y_e, pe, average='weighted'), ext_benign_recall=float((pe[y_e == 0] == 0).mean()),
                       ext_sqli_recall=float((pe[y_e == 1] == 1).mean()), ext_xss_recall=float((pe[y_e == 2] == 2).mean()))
            for pn, g in probes.items():
                lg = logits_of(model, [g], ea, DEVICE)
                row[f'{pn}_pred'] = 'BSX'[int(lg.argmax())]
                row[f'{pn}_margin'] = margin(lg, 1).item()
            raw, rel = sem_attention(model.cpu(), probes['P_split'])
            row.update(attn_raw_ratio=raw, attn_rel_sem=rel)
            model.to(DEVICE)
            if key == 'gat_semW':
                for cond in ('sem_as_seq', 'drop_sem'):     # route E_sem back through W_l / delete the edges
                    lg = logits_of(model, [apply_condition(g, cond) for g in m_graphs], True, DEVICE)
                    c = cell_correct(df_m, lg.argmax(1).numpy())
                    row[f'{cond}_cs_correct'] = c[CS]
                    row[f'{cond}_cs_margin'] = margin(lg[cs_mask], 1).mean().item()
                    row[f'{cond}_cm_correct'] = c[CM]
            row.update({f"cell:{k}": v for k, v in cells.items()})
            rows.append(row)
            print(f"[{key} seed {seed}] test={row['test_acc']:.4f} heldout={row['heldout_correct']}/90 cs={row['cs_correct']}/10 "
                  f"margin={row['cs_margin']:+.2f} extwF1={row['ext_wF1']:.3f} benign_rec={row['ext_benign_recall']:.3f} "
                  f"attn_rel={rel:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / 'results' / 'gat_sem_w_5seed.csv', index=False)
    num = ['test_acc', 'heldout_correct', 'cs_correct', 'cm_correct', 'cs_margin', 'ext_wF1', 'ext_benign_recall',
           'attn_raw_ratio', 'attn_rel_sem']
    agg = out.groupby('model', sort=False)[num].agg(['mean', 'std'])
    agg.insert(0, 'n_seeds', out.groupby('model', sort=False).size())
    passes = out.assign(ok=out.cs_correct == 10).groupby('model', sort=False)['ok'].sum()
    L = ["5-seed mean/std (cs_correct = of 10 rows per seed):", agg.round(3).to_string(), "",
         "comment_splitting all-10-pass seeds: " + ", ".join(f"{k}={v}/{(out.model == k).sum()}" for k, v in passes.items()), ""]
    sem = out[out.model == 'gat_semW']
    others = out[out.model.isin(['gatv2', 'gat_placebo'])]
    if len(sem):
        k1, n1 = int((sem.cs_correct == 10).sum()), len(sem)
        k0, n0 = int((others.cs_correct == 10).sum()), len(others)
        L.append(f"Fisher one-sided, semW {k1}/{n1} vs (gatv2+placebo) {k0}/{n0}: p = {fisher_exact([[k1, n1 - k1], [k0, n0 - k0]], alternative='greater')[1]:.4f}")
        L += ["", "edits on trained gat_semW (mean over seeds): "
              + ", ".join(f"{c}: cs_correct={sem[f'{c}_cs_correct'].mean():.1f}/10 margin={sem[f'{c}_cs_margin'].mean():+.2f}" for c in ('sem_as_seq', 'drop_sem'))]
    L += ["", "per-seed:", out[['model', 'seed'] + num + ['P_split_pred', 'P_split_margin', 'P_intact_pred']].round(3).to_string(index=False)]
    report = "\n".join(L)
    print("\n" + report)
    (ROOT / 'data' / 'gat_sem_w_report.txt').write_text(report + "\n")


if __name__ == '__main__':
    main()
