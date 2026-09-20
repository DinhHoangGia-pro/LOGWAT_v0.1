"""LaTeX rows of the paper's Table 2 (tab:main_results): 11 methods x (test split, held-out matrix, external dataset, latency).

Nothing is typed in; every entry is computed from the results files and cross-checked against results/table2_old_vs_new.csv:
  results/sequence_baselines_5seed.csv, graph_baselines_5seed.csv, gatv2_reference_5seed.csv   5 seeds (42-46) -> mean +- sample std
  results/final_baseline_comparison.csv       RoBERTa / CodeBERT (single run, seed 42) and string matching (deterministic; 'no rule fired' -> Benign in all three modes)
  results/latency_breakdown_graph_baselines.csv   CPU end-to-end mean, GATv2 measured in the same run as GCN/GraphSAGE/GIN/HGT
  results/latency_breakdown.csv               CPU end-to-end mean of the run that also holds LOGWAT's own breakdown (2.01 ms)
Usage: PYTHONPATH=. python -m scripts.build_table2_latex [out.tex]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent.parent / 'results'


def ms(v, nd, scale=1.0):
    v = np.asarray(v, dtype=float) * scale
    return v.mean(), v.std(ddof=1)


def main():
    seq = pd.read_csv(R / 'sequence_baselines_5seed.csv')
    gra = pd.read_csv(R / 'graph_baselines_5seed.csv')
    ref = pd.read_csv(R / 'gatv2_reference_5seed.csv')
    fin = pd.read_csv(R / 'final_baseline_comparison.csv')
    lat_g = pd.read_csv(R / 'latency_breakdown_graph_baselines.csv')
    lat_m = pd.read_csv(R / 'latency_breakdown.csv')
    old = pd.read_csv(R / 'table2_old_vs_new.csv').set_index('method')

    five = {'TextCNN': seq, 'Bi-LSTM': seq, 'Stack-LSTM': seq, 'GCN': gra, 'GraphSAGE': gra, 'GIN': gra, 'HGT': gra, 'LOGWAT (GATv2)': ref}
    key = {'TextCNN': 'TextCNN', 'Bi-LSTM': 'Bi-LSTM', 'Stack-LSTM': 'StackLSTM', 'GCN': 'GCN', 'GraphSAGE': 'GraphSAGE', 'GIN': 'GIN',
           'HGT': 'HGT', 'LOGWAT (GATv2)': 'GATv2'}
    lat_key = {'GCN': 'GCN', 'GraphSAGE': 'GraphSAGE', 'GIN': 'GIN', 'HGT': 'HGT', 'LOGWAT (GATv2)': 'GATv2'}
    old_key = {'LOGWAT (GATv2)': 'Proposed (LOGWAT)', 'TextCNN': 'TextCNN', 'Bi-LSTM': 'Bi-LSTM', 'Stack-LSTM': 'Stack-LSTM', 'GCN': 'GCN',
               'GraphSAGE': 'GraphSAGE', 'GIN': 'GIN', 'HGT': 'HGT'}

    rows = {}
    for name, df in five.items():
        d = df[df.method == key[name]].sort_values('seed')
        assert d.seed.tolist() == [42, 43, 44, 45, 46], (name, d.seed.tolist())
        t = ms(d.test_acc, 3, 100)
        h = ms(d.heldout_correct, 1)
        e = ms(d.ext_weighted_f1, 3)
        # cross-check against the committed 5-seed summary
        o = old.loc[old_key[name]]
        assert abs(t[0] - o.new_test_acc_5s_mean_pct) < 6e-4 and abs(h[0] - o.heldout_correct_5s_mean_of_90) < 0.06, name
        assert abs(e[0] - o.ext_wf1_5s_mean) < 6e-4 and abs(e[1] - o.ext_wf1_5s_std) < 6e-4, (name, e, o.ext_wf1_5s_mean, o.ext_wf1_5s_std)
        lat = None
        if name in lat_key:
            x = lat_g[(lat_g.model == lat_key[name]) & (lat_g.device == 'cpu') & (lat_g.stage == 'end_to_end_ms')]
            lat = float(x.mean_ms.iloc[0])
        rows[name] = dict(test=f"{t[0]:.3f} $\\pm$ {t[1]:.3f}", ho=f"{h[0]:.1f} $\\pm$ {h[1]:.1f}", ext=f"{e[0]:.3f} $\\pm$ {e[1]:.3f}",
                          lat=lat, ho_v=h[0], ext_v=e[0], mark='')

    def single(method, mark):
        f = fin[fin.method == method]
        tw = f[(f.eval_mode == 'test_split') & (f.group == 'weighted avg')].recall.iloc[0] * 100      # weighted recall = accuracy
        h = f[f.eval_mode == 'held_out_matrix']
        ho = float(h.n_correct.sum())
        assert h.n_total.sum() == 90
        ext = f[(f.eval_mode == 'external_dataset') & (f.group == 'weighted avg')].f1_score.iloc[0]
        return dict(test=f"{tw:.3f}", ho=f"{ho:.0f}", ext=f"{ext:.3f}", ho_v=ho, ext_v=ext, mark=mark)

    for name, m in (('RoBERTa', 'RoBERTa'), ('CodeBERT', 'CodeBERT')):
        rows[name] = single(m, '$^{\\dagger}$')
        x = lat_m[(lat_m.method == m) & (lat_m.device == 'cpu') & (lat_m.stage == 'end_to_end_ms')]
        rows[name]['lat'] = float(x.mean_ms.iloc[0])
        rows[name]['lat_sep'] = True
    rows['String matching'] = single('string_matching', '$^{\\ddagger}$')
    rows['String matching']['lat'] = None
    logwat_lat_sep = float(lat_m[(lat_m.method == 'GATv2') & (lat_m.device == 'cpu') & (lat_m.stage == 'end_to_end_ms')].mean_ms.iloc[0])

    trained = ['TextCNN', 'Bi-LSTM', 'Stack-LSTM', 'GCN', 'GraphSAGE', 'GIN', 'HGT', 'LOGWAT (GATv2)']
    best_ho = max(r['ho_v'] for r in rows.values())     # best over ALL methods, including the rule and the transformers
    best_ext = max(r['ext_v'] for r in rows.values())
    lats = {m: r['lat'] for m, r in rows.items() if r.get('lat') is not None and not r.get('lat_sep')}
    best_lat = min(lats.values())

    def cell(txt, bold):
        return f"\\textbf{{{txt}}}" if bold else txt

    def line(name, label=None):
        r = rows[name]
        ho_txt = r['ho'] if r['mark'] == '' else r['ho']
        lat = '--' if r['lat'] is None else f"{r['lat']:.2f}" + ('$^{\\S}$' if r.get('lat_sep') else '')
        lat = cell(lat, r['lat'] is not None and not r.get('lat_sep') and abs(r['lat'] - best_lat) < 1e-9)
        return (f"{label or name}{r['mark']} & {r['test']} & {cell(ho_txt, abs(r['ho_v'] - best_ho) < 1e-9)} & "
                f"{cell(r['ext'], abs(r['ext_v'] - best_ext) < 1e-9)} & {lat} \\\\")

    L = []
    L.append("\\begin{tabular}{|l|c|c|c|c|}")
    L.append("\\hline")
    L.append("\\textbf{Method} & \\textbf{Test (\\%)} & \\textbf{Held-out (of 90)} & \\textbf{External wF1} & \\textbf{Latency (ms)} \\\\")
    L.append("\\hline")
    L.append(line('String matching'))
    L.append("\\hline")
    for m in ('TextCNN', 'Bi-LSTM', 'Stack-LSTM'):
        L.append(line(m))
    L.append("\\hline")
    for m in ('RoBERTa', 'CodeBERT'):
        L.append(line(m))
    L.append("\\hline")
    for m in ('GCN', 'GraphSAGE', 'GIN'):
        L.append(line(m))
    L.append(line('HGT', 'HGT (typed edges)'))
    L.append("\\hline")
    L.append(line('LOGWAT (GATv2)', '\\textbf{LOGWAT (GATv2)}'))
    L.append("\\hline")
    L.append("\\end{tabular}")
    out = "\n".join(L)
    print(out)
    print(f"\n% latency of LOGWAT in the separate run that also holds the transformers: {logwat_lat_sep:.2f} ms", file=sys.stderr)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(out + "\n")


if __name__ == '__main__':
    main()
