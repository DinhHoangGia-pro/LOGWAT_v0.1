"""Add TRAIN-only rows for the 16 SQLi payloads (pool index 25-40: the last
7 Error-based + all 9 Stacked/other) that are present in
data/sqli_payload_pool.csv but were never incorporated into the frozen
data/augmented_web_attack.csv (generated when the pool only had 25 rows).

Deliberately NOT a full regenerate of the dataset via prepare_data.py: a
dry-run comparison showed that re-running balance_dataset() with the
current 41-row pool keeps all Benign/XSS/csic_original rows byte-identical
and positionally stable, but reassigns SOURCE_UID for ALL 41 SQLi payload
groups relative to the frozen data/test_split_indices.pkl -- every payload
would end up split across both the old train and old test positions,
reintroducing exactly the train/test content leakage bug that was already
fixed once (grouped GroupShuffleSplit on source_uid). Confirmed with real
data: 41/41 payload groups mixed, 41/41 payload content strings appearing
in both old-train and old-test positions.

This script instead appends brand-new rows (poisoning benign frames the
same way balance_dataset() does), with source_uid values that can never
collide with, or be swept into, any existing split (global test_idx in
test_split_indices.pkl, or the SQLi family-test uids computed by
src.training.train._fixed_family_group_split). Existing rows are never
rewritten.
"""
import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
AUGMENTED_CSV = DATA_DIR / 'augmented_web_attack.csv'
POOL_CSV = DATA_DIR / 'sqli_payload_pool.csv'
REPORT_PATH = DATA_DIR / 'missing_sqli_payloads_train_report.txt'

ROWS_PER_PAYLOAD = 400  # matches the density of the 25 payloads already present (10000/25)


def main():
    df = pd.read_csv(AUGMENTED_CSV)
    n_before = len(df)

    if df['source'].eq('sqli_pool_extra').any():
        raise RuntimeError(
            "Dữ liệu nguồn 'sqli_pool_extra' đã tồn tại trong augmented_web_attack.csv "
            "— dừng để tránh append trùng. Xoá thủ công nếu thực sự muốn chạy lại."
        )

    pool = pd.read_csv(POOL_CSV)
    present_uids = set(df[df['source'] == 'sqli_pool']['source_uid'].dropna().astype(str))
    present_indices = {int(u.split('_')[-1]) for u in present_uids}
    missing_indices = sorted(set(range(len(pool))) - present_indices)

    benign_frames = df[df['attack_type'] == 0].reset_index(drop=True)
    if len(benign_frames) == 0:
        raise RuntimeError('No benign frames available as row templates')

    new_rows = []
    seq = 0
    for payload_idx in missing_indices:
        payload = str(pool.iloc[payload_idx]['payload'])
        family = str(pool.iloc[payload_idx]['family'])
        for _ in range(ROWS_PER_PAYLOAD):
            template = benign_frames.iloc[seq % len(benign_frames)].to_dict()
            template['content'] = payload
            template['attack_type'] = 1
            template['source'] = 'sqli_pool_extra'
            template['source_uid'] = f'sqli_pool_extra_{payload_idx}_{seq}'
            new_rows.append(template)
            seq += 1

    new_df = pd.DataFrame(new_rows).reindex(columns=df.columns)

    # Append-only write: never re-serialize existing rows (a prior mistake
    # this session showed that a read+rewrite round trip re-escapes already
    # backslash-escaped content).
    with open(AUGMENTED_CSV, 'a', newline='') as f:
        new_df.to_csv(f, index=False, header=False, quoting=csv.QUOTE_ALL, escapechar='\\')

    n_after = n_before + len(new_df)

    lines = []
    lines.append("=== Train-only addition: missing SQLi payload pool indices 25-40 ===")
    lines.append(f"sqli_payload_pool.csv total payloads: {len(pool)}")
    lines.append(f"payload indices already present in augmented_web_attack.csv: {sorted(present_indices)}")
    lines.append(f"missing indices added by this script: {missing_indices}")
    lines.append("")
    for idx in missing_indices:
        lines.append(f"  idx={idx} family={pool.iloc[idx]['family']!r} payload={pool.iloc[idx]['payload']!r}")
    lines.append("")
    lines.append(f"rows per payload: {ROWS_PER_PAYLOAD} (matches density of the 25 payloads already present)")
    lines.append(f"new rows appended: {len(new_df)}")
    lines.append(f"dataset rows before: {n_before}")
    lines.append(f"dataset rows after:  {n_after}")
    lines.append("")
    lines.append("source='sqli_pool_extra', source_uid='sqli_pool_extra_<payload_idx>_<seq>' -- new values,")
    lines.append("cannot collide with any existing source_uid, and (being appended past the frozen")
    lines.append("30084/32195-row range) automatically fall outside data/test_split_indices.pkl's test_idx.")
    report = "\n".join(lines)
    REPORT_PATH.write_text(report + "\n")
    print(report)


if __name__ == '__main__':
    main()
