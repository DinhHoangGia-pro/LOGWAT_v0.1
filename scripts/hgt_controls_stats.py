"""Significance tests on results/hgt_controls_5seed.csv (typed HGT vs the two same-architecture controls).

Fisher exact (one-sided) on "solves comment_splitting" and "collapses on external Benign"; Mann-Whitney on
external weighted F1. n is tiny (5 typed vs 10 control runs): read as indicative. Also checks that
hgt_random and GATv2 really are different models (their probe-count coincidence is not a loading bug).

Writes results/hgt_controls_tests.txt.
"""
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

from scripts.hgt_investigation_common import ROOT, SEEDS, load_model, logits_of, uses_edge_attr
from scripts.investigate_hgt_generalization import PROBES, PROBE_LABEL
from src.bag.graph_builder import build_single_graph


def main():
    d = pd.read_csv(ROOT / 'results' / 'hgt_controls_5seed.csv')
    ctrl = d[d.model.isin(['hgt_collapsed', 'hgt_random'])]
    typ = d[d.model == 'hgt']
    F = lambda t: fisher_exact(t, alternative='greater')[1]
    L = []
    k = int((ctrl.cs_correct == 10).sum())
    L.append(f"comment_splitting all-10-pass: typed 5/5 vs controls {k}/10 ; Fisher one-sided p = {F([[5, 0], [k, 10 - k]]):.4f}")
    c = lambda x: int((x.ext_benign_recall < 0.2).sum())
    L.append(f"external Benign collapse (recall<0.2): typed {c(typ)}/5 vs controls {c(ctrl)}/10 ; "
             f"Fisher one-sided p = {F([[c(typ), 5 - c(typ)], [c(ctrl), 10 - c(ctrl)]]):.4f}")
    L.append("held-out correct/90  typed " + str(typ.heldout_correct.tolist()) +
             " | collapsed " + str(d[d.model == 'hgt_collapsed'].heldout_correct.tolist()) +
             " | random " + str(d[d.model == 'hgt_random'].heldout_correct.tolist()))
    L.append(f"ext wF1 typed vs pooled controls Mann-Whitney p (two-sided) = {mannwhitneyu(typ.ext_wF1, ctrl.ext_wF1).pvalue:.3f}")
    gs = [build_single_graph(s, source_uid='p', attack_type=PROBE_LABEL[g], use_edge_attr=True)
          for g, v in PROBES.items() for s in v]
    P = {m: np.stack([logits_of(load_model(m, s), gs, uses_edge_attr(m)).argmax(1).numpy() for s in SEEDS])
         for m in ('gatv2', 'hgt_random')}
    L.append(f"probe predictions gatv2 vs hgt_random: fraction equal over 5 seeds x 30 strings = {(P['gatv2'] == P['hgt_random']).mean():.3f} "
             "(<1 => different models)")
    print("\n".join(L))
    (ROOT / 'results' / 'hgt_controls_tests.txt').write_text("\n".join(L) + "\n")


if __name__ == '__main__':
    main()
