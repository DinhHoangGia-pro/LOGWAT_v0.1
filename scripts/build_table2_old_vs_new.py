"""Old Table 2 (paper draft) vs the #16 re-run, per baseline -- the RQF-06 evidence.

OLD numbers are transcribed verbatim from Table `tab:method_comparison_extended`
in hin_web_vulne/sn-article_v1.tex (measured on the ORIGINAL 30k-row dataset/split,
with baseline configurations for which no code or hyperparameters survive in this
repo -- so they cannot be re-derived, only compared against).
NEW numbers are the #16 re-run on the CURRENT dataset (43,659 rows) and frozen split,
all through the same train.py loop / eval code as GATv2:
  new_test_acc_s42      seed-42 test-split accuracy (= 'weighted avg' recall)
  new_test_acc_5s_mean  mean over seeds 42..46 (results/*_5seed.csv)
  delta_test_pp         new_test_acc_5s_mean - old_acc, in percentage points
  heldout_cells_5s      mean of fully-correct cells (out of 9) over 5 seeds
  ext_wf1_5s_mean/std   external-dataset weighted F1, mean/std over 5 seeds

Writes results/table2_old_vs_new.csv.
"""
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / 'results'

# (method as in results/*.csv, old label in Table 2, old Acc %, old F1)
OLD = [
    ('TextCNN', 'TextCNN', 89.54, 0.87),
    ('Bi-LSTM', 'Bi-LSTM', 91.20, 0.89),
    ('StackLSTM', 'Stack-LSTM', 92.15, 0.91),
    ('GCN', 'GCN', 93.45, 0.92),
    ('GraphSAGE', 'GraphSAGE', 93.88, 0.92),
    ('GIN', 'GIN', 94.12, 0.93),
    (None, 'GATv2 (vanilla)', 95.30, 0.94),   # no counterpart re-run: 'vanilla' is not defined in this repo
    ('HGT', 'HGT', 95.87, 0.95),
    ('GATv2', 'Proposed (LOGWAT)', 97.84, 0.98),
]


def main():
    seed42 = pd.concat([pd.read_csv(RESULTS / n) for n in ('graph_baselines.csv', 'sequence_baselines.csv')])
    five = pd.concat([pd.read_csv(RESULTS / n) for n in
                      ('graph_baselines_5seed.csv', 'sequence_baselines_5seed.csv', 'gatv2_reference_5seed.csv')])
    final = pd.read_csv(RESULTS / 'final_baseline_comparison.csv')
    rows = []
    for method, label, old_acc, old_f1 in OLD:
        row = {'method': label, 'old_acc_pct': old_acc, 'old_f1': old_f1}
        if method is not None:
            src = final[(final['method'] == method) & (final['eval_mode'] == 'test_split') & (final['group'] == 'weighted avg')]
            row['new_test_acc_s42_pct'] = round(100 * float(src['recall'].iloc[0]), 2)
            f5 = five[five['method'] == method]
            row['n_seeds'] = len(f5)
            row['new_test_acc_5s_mean_pct'] = round(100 * f5['test_acc'].mean(), 3)
            row['delta_test_pp'] = round(row['new_test_acc_5s_mean_pct'] - old_acc, 2)
            row['heldout_cells_5s_mean'] = round(f5['heldout_full_cells'].mean(), 2)
            row['heldout_correct_5s_mean_of_90'] = round(f5['heldout_correct'].mean(), 1)
            row['ext_wf1_5s_mean'] = round(f5['ext_weighted_f1'].mean(), 3)
            row['ext_wf1_5s_std'] = round(f5['ext_weighted_f1'].std(), 3)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / 'table2_old_vs_new.csv', index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
