"""String-matching baseline (src.preprocessing.normalization.classify_request)
on the frozen test split -- the one evaluation mode it didn't already have a
result for (held-out matrix: results/heldout_matrix_full.csv; external:
results/external_dataset_evaluation.csv already include it). Needed to
complete the 4-method (GATv2/RoBERTa/CodeBERT/string-matching) comparison
across all 3 evaluation modes.

Same -1/3 (unknown/other) fallback-to-Benign(0) treatment already used for
this baseline elsewhere (scripts/build_heldout_matrix_eval.py,
scripts/evaluate_external_dataset.py), for consistency.

Writes results/string_matching_test_split.csv.
"""
import shutil
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from src.preprocessing.normalization import classify_request

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
TEST_CSV = DATA_DIR / 'transformer_baseline' / 'test.csv'
OUT_CSV = RESULTS_DIR / 'string_matching_test_split.csv'
CLASS_NAMES = ['Benign', 'SQLi', 'XSS']


def main():
    df = pd.read_csv(TEST_CSV)
    y_true = df['attack_type'].astype(int).tolist()
    y_pred = []
    for content in df['content'].astype(str).tolist():
        pred = classify_request(content, None)
        y_pred.append(int(pred) if pred in (0, 1, 2) else 0)

    report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
                                    output_dict=True, zero_division=0)
    rows = []
    for key, metrics in report.items():
        if not isinstance(metrics, dict):
            continue
        rows.append({'method': 'string_matching', 'label': key,
                      'precision': metrics.get('precision'), 'recall': metrics.get('recall'),
                      'f1_score': metrics.get('f1-score'), 'support': metrics.get('support')})

    if OUT_CSV.exists():
        backup = RESULTS_DIR / f"string_matching_test_split_PRE_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        shutil.copy(OUT_CSV, backup)
        print(f"[*] backed up existing {OUT_CSV} -> {backup}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT_CSV, index=False)
    print(f"n={len(df)}")
    print(result_df.to_string(index=False))
    print(f"\nWrote: {OUT_CSV}")


if __name__ == '__main__':
    main()
