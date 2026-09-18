"""Clean/normalize the external evaluation dataset (HttpParamsDataset) into
the same 3-class (0=Benign, 1=SQLi, 2=XSS) format used by
data/augmented_web_attack.csv, for EVALUATION ONLY (never appended to the
training data, never used to fit anything).

Source columns: payload, length, attack_type (norm/sqli/xss/cmdi/path-traversal),
label (norm/anom). cmdi and path-traversal rows are dropped -- out of scope
for this repo's 3-class model.

Also prints (and saves) the per-class average `content` length compared
against data/augmented_web_attack.csv's `content` column, per the task's
explicit domain-shift-in-format check: the model only ever sees the `content`
column (src/bag/graph_builder.py:build_web_graphs reads row['content'] and
nothing else), so this is the correct, directly comparable field.
"""
from pathlib import Path

import pandas as pd

from src.preprocessing.normalization import clean_content

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
EXTERNAL_DIR = DATA_DIR / 'external'
RAW_CSV = EXTERNAL_DIR / 'httpparamsdataset_raw.csv'
CLEAN_CSV = EXTERNAL_DIR / 'external_dataset_clean.csv'
AUGMENTED_CSV = DATA_DIR / 'augmented_web_attack.csv'
REPORT_PATH = EXTERNAL_DIR / 'external_dataset_prep_report.txt'

LABEL_MAP = {'norm': 0, 'sqli': 1, 'xss': 2}  # cmdi/path-traversal dropped
CLASS_NAMES = {0: 'Benign', 1: 'SQLi', 2: 'XSS'}


def main():
    raw = pd.read_csv(RAW_CSV)
    n_raw = len(raw)

    raw = raw[raw['attack_type'].isin(LABEL_MAP.keys())].copy()
    n_after_label_filter = len(raw)
    dropped_out_of_scope = n_raw - n_after_label_filter

    raw['attack_type_mapped'] = raw['attack_type'].map(LABEL_MAP)
    raw['content'] = raw['payload'].apply(clean_content)
    raw = raw.dropna(subset=['content'])
    n_after_clean = len(raw)

    raw = raw.reset_index(drop=True)
    raw['source'] = 'external_httpparamsdataset'
    raw['source_uid'] = [f'external_httpparamsdataset_{i}' for i in range(len(raw))]

    out = pd.DataFrame({
        'content': raw['content'],
        'attack_type': raw['attack_type_mapped'].astype(int),
        'source': raw['source'],
        'source_uid': raw['source_uid'],
        'orig_attack_type_label': raw['attack_type'],
    })

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CLEAN_CSV, index=False)

    class_counts = out['attack_type'].value_counts().sort_index()

    # --- Domain-shift-in-format check: avg content length, per class, vs train ---
    train = pd.read_csv(AUGMENTED_CSV)
    train_len_by_class = train.assign(_len=train['content'].astype(str).str.len()).groupby('attack_type')['_len'].mean()
    ext_len_by_class = out.assign(_len=out['content'].astype(str).str.len()).groupby('attack_type')['_len'].mean()

    lines = []
    lines.append("=== External dataset (HttpParamsDataset) prep report ===")
    lines.append(f"Raw rows: {n_raw}")
    lines.append(f"Dropped (attack_type not in norm/sqli/xss, i.e. cmdi/path-traversal): {dropped_out_of_scope}")
    lines.append(f"After label filter: {n_after_label_filter}")
    lines.append(f"After clean_content() (URL-unquote, drop len<2/NaN): {n_after_clean}")
    lines.append(f"Rows written to {CLEAN_CSV.name}: {len(out)}")
    lines.append("")
    lines.append("Class counts (0=Benign, 1=SQLi, 2=XSS):")
    for cls, cnt in class_counts.items():
        lines.append(f"  {CLASS_NAMES[cls]} ({cls}): {cnt}")
    lines.append("")
    lines.append("=== Avg `content` length (chars), per class: external vs train (augmented_web_attack.csv) ===")
    any_over_50pct = False
    for cls in (0, 1, 2):
        train_avg = float(train_len_by_class.get(cls, float('nan')))
        ext_avg = float(ext_len_by_class.get(cls, float('nan')))
        if train_avg and ext_avg == ext_avg:  # not NaN
            pct_diff = (ext_avg - train_avg) / train_avg * 100
        else:
            pct_diff = float('nan')
        flag = ''
        if abs(pct_diff) > 50:
            flag = '  <<< >50% DIFFERENCE -- FORMAT DOMAIN SHIFT'
            any_over_50pct = True
        lines.append(f"  {CLASS_NAMES[cls]}: train avg={train_avg:.1f} chars, external avg={ext_avg:.1f} chars, "
                     f"diff={pct_diff:+.1f}%{flag}")

    lines.append("")
    if any_over_50pct:
        lines.append("CONCLUSION: at least one class has a >50% average-length difference between "
                     "external and train `content`. The external set is made of short, isolated HTTP "
                     "parameter VALUES (single form fields / payload fragments), not full HTTP request "
                     "strings like augmented_web_attack.csv's `content` column. Step 4 results must be "
                     "read as testing generalization to a different INPUT FORMAT, not just different "
                     "attack content -- this is a confound, not a clean apples-to-apples generalization test.")
    else:
        lines.append("CONCLUSION: no class shows a >50% average-length difference; format is reasonably comparable.")

    report = "\n".join(lines)
    REPORT_PATH.write_text(report + "\n")
    print(report)
    print(f"\nWrote: {CLEAN_CSV}")
    print(f"Wrote: {REPORT_PATH}")


if __name__ == '__main__':
    main()
