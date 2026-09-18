"""Evaluate the fine-tuned RoBERTa/CodeBERT baselines (Reviewer #3) on the
SAME three tests already run for GATv2, for an across-the-board comparison
instead of one number:

  a) frozen test split (data/transformer_baseline/test.csv, same rows as
     GATv2's test_idx -- see scripts/prepare_transformer_baseline_data.py)
  b) held-out 9-cell matrix (scripts/build_heldout_matrix_eval.py's
     make_matrix_rows(), reused directly so the exact same 90 rows are used)
  c) external dataset (data/external/external_dataset_clean.csv, text form
     -- NOT the graph .pkl, which is GATv2-specific)

Baselines are text classifiers -- no graph construction, no edge ablation.
A baseline that fails the held-out matrix (e.g. near-0 accuracy on cells
designed to defeat literal keyword matching) is itself a meaningful,
expected result here, not a bug to fix.

Also breaks the test-split result down by whether each row's `content` is
byte-identical to some row in train (data/transformer_baseline/train.csv) --
see docs/EXPERIMENT_LOG_semantic_edge_investigation.md's "Test-split content
duplication (25.2%, primarily XSS 45.7%)" section: 25.2% of the frozen test
split shares exact content with train, a property of the frozen split
itself, not this baseline. Reported here on each baseline's FINAL
checkpoint (not a one-off probe) as direct evidence the baseline is not
unfairly benefiting from it, not just a caveat asserted without numbers.

Writes results/transformer_baselines.csv, long format:
  method, eval_mode, group, precision, recall, f1_score, support,
  accuracy, n_correct, n_total
(precision/recall/f1_score/support populated for eval_mode in
{test_split, external_dataset}; accuracy/n_correct/n_total populated for
eval_mode in {held_out_matrix, test_split_duplication_breakdown}; the other
columns are blank per row.)

Backs up any existing results/transformer_baselines.csv to
results/transformer_baselines_PRE_<label>.csv before overwriting.
"""
import argparse
import shutil
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from scripts.build_heldout_matrix_eval import make_matrix_rows, TECHNIQUES, LABELS
from src.baselines.transformer_common import CLASS_NAMES, load_finetuned, predict_batch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
TRAIN_CSV = DATA_DIR / 'transformer_baseline' / 'train.csv'
TEST_CSV = DATA_DIR / 'transformer_baseline' / 'test.csv'
EXTERNAL_CSV = DATA_DIR / 'external' / 'external_dataset_clean.csv'
OUT_CSV = RESULTS_DIR / 'transformer_baselines.csv'

METHODS = {
    'RoBERTa': DATA_DIR / 'models_pretrained' / 'roberta_baseline_seed42',
    'CodeBERT': DATA_DIR / 'models_pretrained' / 'codebert_baseline_seed42',
}


def classification_report_rows(method, eval_mode, y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
                                    output_dict=True, zero_division=0)
    rows = []
    for key, metrics in report.items():
        if not isinstance(metrics, dict):
            continue
        rows.append({
            'method': method, 'eval_mode': eval_mode, 'group': key,
            'precision': metrics.get('precision'), 'recall': metrics.get('recall'),
            'f1_score': metrics.get('f1-score'), 'support': metrics.get('support'),
            'accuracy': None, 'n_correct': None, 'n_total': None,
        })
    return rows


def evaluate_test_split(method, model, tokenizer, max_length, device):
    df = pd.read_csv(TEST_CSV)
    preds, _ = predict_batch(model, tokenizer, df['content'].astype(str).tolist(), max_length, device)
    y_true = df['attack_type'].astype(int).tolist()
    print(f"[{method}][test_split] n={len(df)}")
    rows = classification_report_rows(method, 'test_split', y_true, preds)
    rows.extend(duplication_breakdown_rows(method, df, y_true, preds))
    return rows


def duplication_breakdown_rows(method, test_df, y_true, y_pred):
    """Test-split accuracy split by whether each row's content is
    byte-identical to some train row -- run on the FINAL checkpoint (called
    from evaluate_test_split, which already has the full-test-set
    predictions), not a one-off epoch-1 probe."""
    train_contents = set(pd.read_csv(TRAIN_CSV)['content'].astype(str))
    dup_mask = test_df['content'].astype(str).isin(train_contents).tolist()

    rows = []
    for flag, group in [(True, 'duplicate_content'), (False, 'novel_content')]:
        idxs = [i for i, d in enumerate(dup_mask) if d == flag]
        if not idxs:
            continue
        yt = [y_true[i] for i in idxs]
        yp = [y_pred[i] for i in idxs]
        n_correct = sum(1 for a, b in zip(yt, yp) if a == b)
        rows.append({
            'method': method, 'eval_mode': 'test_split_duplication_breakdown', 'group': group,
            'precision': None, 'recall': None, 'f1_score': None, 'support': None,
            'accuracy': n_correct / len(idxs), 'n_correct': n_correct, 'n_total': len(idxs),
        })
    print(f"[{method}][test_split_duplication_breakdown] " +
          ", ".join(f"{r['group']}={r['n_correct']}/{r['n_total']} ({r['accuracy']:.4f})" for r in rows))
    return rows


def evaluate_heldout_matrix(method, model, tokenizer, max_length, device):
    df = make_matrix_rows()
    rows = []
    for label in [0, 1, 2]:
        for technique in TECHNIQUES[label]:
            subset = df[(df['class_name'] == LABELS[label]) & (df['technique'] == technique)]
            if subset.empty:
                continue
            y_true = subset['attack_type'].astype(int).tolist()
            preds, _ = predict_batch(model, tokenizer, subset['content'].astype(str).tolist(), max_length, device)
            n_correct = sum(1 for a, b in zip(y_true, preds) if a == b)
            rows.append({
                'method': method, 'eval_mode': 'held_out_matrix',
                'group': f"{LABELS[label]}/{technique}",
                'precision': None, 'recall': None, 'f1_score': None, 'support': None,
                'accuracy': n_correct / len(subset), 'n_correct': n_correct, 'n_total': len(subset),
            })
    print(f"[{method}][held_out_matrix] {sum(r['n_correct'] for r in rows)}/{sum(r['n_total'] for r in rows)} cells-correct-total")
    return rows


def evaluate_external(method, model, tokenizer, max_length, device):
    df = pd.read_csv(EXTERNAL_CSV)
    preds, _ = predict_batch(model, tokenizer, df['content'].astype(str).tolist(), max_length, device)
    y_true = df['attack_type'].astype(int).tolist()
    print(f"[{method}][external_dataset] n={len(df)}")
    return classification_report_rows(method, 'external_dataset', y_true, preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='*', default=list(METHODS.keys()))
    args = parser.parse_args()

    if OUT_CSV.exists():
        backup = RESULTS_DIR / f"transformer_baselines_PRE_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        shutil.copy(OUT_CSV, backup)
        print(f"[*] backed up existing {OUT_CSV} -> {backup}")

    all_rows = []
    for method in args.methods:
        output_dir = METHODS[method]
        if not output_dir.exists():
            print(f"[!] {method} checkpoint not found at {output_dir}, skipping")
            continue
        model, tokenizer, max_length, device = load_finetuned(output_dir)
        all_rows.extend(evaluate_test_split(method, model, tokenizer, max_length, device))
        all_rows.extend(evaluate_heldout_matrix(method, model, tokenizer, max_length, device))
        all_rows.extend(evaluate_external(method, model, tokenizer, max_length, device))

    result_df = pd.DataFrame(all_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(result_df)} rows to {OUT_CSV}")
    print(result_df.to_string(index=False))


if __name__ == '__main__':
    main()
