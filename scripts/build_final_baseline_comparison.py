"""Assemble the final comparison -- GATv2 / RoBERTa / CodeBERT / string-matching,
plus (#16) the re-run graph baselines (GCN / GraphSAGE / GIN / HGT) and sequence
baselines (Bi-LSTM / TextCNN / StackLSTM) -- across all 3 evaluation modes (test
split, held-out 9-cell matrix, external dataset). The table intended to replace
Table 2 in the paper (Reviewer #3).

Pulls from results already on disk rather than recomputing anything:
  - GATv2 test split:      results/main_results.csv
                            (subset=='SQLi tổng hợp' & source=='all', the
                            full 3-class report on the whole frozen test set)
  - GATv2 + string-matching held-out matrix: results/heldout_matrix_full.csv
  - GATv2 + string-matching external:  results/external_dataset_evaluation.csv
  - string-matching test split:        results/string_matching_test_split.csv
  - RoBERTa + CodeBERT (all 3 modes):  results/transformer_baselines.csv
  - GCN/GraphSAGE/GIN/HGT (all 3 modes):    results/graph_baselines.csv
  - Bi-LSTM/TextCNN/StackLSTM (all 3 modes): results/sequence_baselines.csv
    (both in the same schema as transformer_baselines.csv; skipped, with a
    message, if the file doesn't exist yet)

Writes results/final_baseline_comparison.csv, long format:
  method, eval_mode, group, precision, recall, f1_score, support,
  accuracy, n_correct, n_total
(same schema as results/transformer_baselines.csv, so the 4 methods sit in
one table with no per-method special-casing downstream.)
"""
import shutil
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'results'
OUT_CSV = RESULTS_DIR / 'final_baseline_comparison.csv'

COLUMNS = ['method', 'eval_mode', 'group', 'precision', 'recall', 'f1_score',
           'support', 'accuracy', 'n_correct', 'n_total']


def blank_row(**kwargs):
    row = {c: None for c in COLUMNS}
    row.update(kwargs)
    return row


def gatv2_test_split_rows():
    df = pd.read_csv(RESULTS_DIR / 'main_results.csv')
    subset = df[(df['subset'] == 'SQLi tổng hợp') & (df['source'] == 'all')]
    rows = []
    for _, r in subset.iterrows():
        rows.append(blank_row(method='GATv2', eval_mode='test_split', group=str(r['label']),
                               precision=r['precision'], recall=r['recall'],
                               f1_score=r['f1_score'], support=r['support']))
    return rows


def string_matching_test_split_rows():
    df = pd.read_csv(RESULTS_DIR / 'string_matching_test_split.csv')
    rows = []
    for _, r in df.iterrows():
        rows.append(blank_row(method='string_matching', eval_mode='test_split', group=str(r['label']),
                               precision=r['precision'], recall=r['recall'],
                               f1_score=r['f1_score'], support=r['support']))
    return rows


def heldout_matrix_rows():
    df = pd.read_csv(RESULTS_DIR / 'heldout_matrix_full.csv')
    rows = []
    for _, r in df.iterrows():
        group = f"{r['class_name']}/{r['technique']}"
        rows.append(blank_row(method='GATv2', eval_mode='held_out_matrix', group=group,
                               accuracy=r['gatv2_accuracy'], n_correct=r['gatv2_correct'],
                               n_total=r['n_samples']))
        rows.append(blank_row(method='string_matching', eval_mode='held_out_matrix', group=group,
                               accuracy=r['string_matching_accuracy'], n_correct=r['string_matching_correct'],
                               n_total=r['n_samples']))
    return rows


def external_rows():
    df = pd.read_csv(RESULTS_DIR / 'external_dataset_evaluation.csv')
    method_map = {'gatv2': 'GATv2', 'string_matching': 'string_matching'}
    rows = []
    for _, r in df.iterrows():
        if r['method'] not in method_map:
            continue  # skip decision_tree_size_only -- not one of the 4 methods in this table
        rows.append(blank_row(method=method_map[r['method']], eval_mode='external_dataset',
                               group=str(r['label']), precision=r['precision'], recall=r['recall'],
                               f1_score=r['f1_score'], support=r['support']))
    return rows


def transformer_rows():
    df = pd.read_csv(RESULTS_DIR / 'transformer_baselines.csv')
    return df[COLUMNS].to_dict('records')


def baseline_csv_rows(name):
    """Rows of a results/<name>.csv already in this table's schema (#16 baselines)."""
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"[!] {name} not found, skipping")
        return []
    return pd.read_csv(path)[COLUMNS].to_dict('records')


def main():
    rows = []
    rows.extend(gatv2_test_split_rows())
    rows.extend(string_matching_test_split_rows())
    rows.extend(heldout_matrix_rows())
    rows.extend(external_rows())
    rows.extend(transformer_rows())
    rows.extend(baseline_csv_rows('graph_baselines.csv'))
    rows.extend(baseline_csv_rows('sequence_baselines.csv'))

    if OUT_CSV.exists():
        backup = RESULTS_DIR / f"final_baseline_comparison_PRE_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        shutil.copy(OUT_CSV, backup)
        print(f"[*] backed up existing {OUT_CSV} -> {backup}")

    result_df = pd.DataFrame(rows, columns=COLUMNS)
    result_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(result_df)} rows to {OUT_CSV}")

    print("\n=== test_split (macro/weighted avg + per-class) ===")
    ts = result_df[result_df['eval_mode'] == 'test_split']
    print(ts.to_string(index=False))

    print("\n=== held_out_matrix (cells-correct summary per method) ===")
    hm = result_df[result_df['eval_mode'] == 'held_out_matrix']
    summary = hm.groupby('method').agg(cells_correct=('n_correct', 'sum'), cells_total=('n_total', 'sum'))
    print(summary.to_string())

    print("\n=== external_dataset (macro/weighted avg per method) ===")
    ext = result_df[(result_df['eval_mode'] == 'external_dataset') & (result_df['group'].isin(['macro avg', 'weighted avg']))]
    print(ext.to_string(index=False))


if __name__ == '__main__':
    main()
