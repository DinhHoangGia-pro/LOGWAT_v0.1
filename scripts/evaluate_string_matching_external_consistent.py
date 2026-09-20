"""String-matching baseline on the external dataset with the SAME convention as the test split and the held-out matrix.

scripts/evaluate_string_matching_test_split.py and scripts/build_heldout_matrix_eval.py map an "unknown" prediction of
classify_request() (no rule fired) to Benign (0), which is what a rule-based WAF does with a request it lets through.
scripts/evaluate_external_dataset.py instead counts unknown as wrong (-1), which gives Benign recall 0 and weighted F1 0.308.
This script recomputes the external numbers with the Benign mapping. It writes a NEW file and touches no existing result.

Writes results/string_matching_external_unknown_as_benign.csv (long schema of final_baseline_comparison.csv, group = class).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src.preprocessing.normalization import classify_request

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results' / 'string_matching_external_unknown_as_benign.csv'


def main():
    df = pd.read_csv(ROOT / 'data' / 'external' / 'external_dataset_clean.csv')
    y = df['attack_type'].astype(int).values
    raw = np.array([classify_request(c) for c in df['content'].astype(str)])
    pred = np.where(np.isin(raw, [0, 1, 2]), raw, 0)
    rep = classification_report(y, pred, labels=[0, 1, 2], target_names=['Benign', 'SQLi', 'XSS'], output_dict=True, zero_division=0)
    rows = []
    for k, m in rep.items():
        if isinstance(m, dict):
            rows.append(dict(method='string_matching', eval_mode='external_dataset_unknown_as_benign', group=k, precision=m['precision'],
                             recall=m['recall'], f1_score=m['f1-score'], support=m['support']))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"unmatched (classify_request == -1 or 3): {int((~np.isin(raw, [0, 1, 2])).sum())} of {len(df)}")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"[+] wrote {OUT}")


if __name__ == '__main__':
    main()
