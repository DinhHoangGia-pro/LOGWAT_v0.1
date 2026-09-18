"""Add comment-splitting-style noise variants to TRAIN-only SQLi rows.

Mechanism is deliberately DIFFERENT from the held-out set's noise
(scripts/build_heldout_matrix_eval.py uses a fixed `/**/` insertion at a
roughly-mid split point inside union/select/from). Here we insert 1-3
whitespace characters at a RANDOM interior position inside a RANDOMLY
chosen SQL keyword occurrence, drawn from a broader keyword list. This
keeps the held-out `comment_splitting` cell a genuine unseen-technique
test instead of training the model on the exact mechanism it's evaluated
against.

Only rows whose source_uid is NOT in data/test_split_indices.pkl's
test_idx AND NOT in the SQLi-family test set (same deterministic rule as
src.training.train._fixed_family_group_split) are eligible. New rows are
appended with a brand-new, globally-unique source_uid/source so they can
never collide with any existing held-out or test-designated group.
"""
import csv
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.training.train import _load_sqli_family_map

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
AUGMENTED_CSV = DATA_DIR / 'augmented_web_attack.csv'
TEST_SPLIT_PATH = DATA_DIR / 'test_split_indices.pkl'
REPORT_PATH = DATA_DIR / 'train_noise_augmentation_report.txt'

NOISE_FRACTION = 0.35  # midpoint of requested 30-40%
SEED = 42

# Broader SQL keyword list than semantic.py's 9-pair E_sem vocabulary,
# so the train-time noise mechanism isn't narrowly scoped to only the
# words exercised by the semantic-edge fix.
SQL_KEYWORDS = [
    'select', 'union', 'from', 'where', 'order', 'group', 'insert', 'into',
    'drop', 'table', 'delete', 'update', 'and', 'or', 'null', 'waitfor',
    'delay', 'benchmark', 'sleep', 'extractvalue', 'updatexml', 'convert',
    'cast', 'database', 'version', 'concat', 'exec', 'all', 'limit',
]
_KEYWORD_RE = re.compile(r"\b(" + "|".join(SQL_KEYWORDS) + r")\b", re.IGNORECASE)
_WHITESPACE_VARIANTS = [' ', '  ', '\t', ' \t ', '   ']


def find_keyword_spans(content: str):
    return [m.span() for m in _KEYWORD_RE.finditer(content)]


def noise_content(content: str, rng: random.Random):
    spans = find_keyword_spans(content)
    if not spans:
        return None
    start, end = rng.choice(spans)
    word = content[start:end]
    if len(word) < 2:
        return None
    split_pos = rng.randint(1, len(word) - 1)
    filler = rng.choice(_WHITESPACE_VARIANTS)
    noised_word = word[:split_pos] + filler + word[split_pos:]
    return content[:start] + noised_word + content[end:]


def compute_family_test_uids():
    family_map, ordered_families = _load_sqli_family_map()
    family_to_uids = defaultdict(list)
    for uid in sorted(family_map.keys(), key=lambda x: int(x.split('_')[-1])):
        family_to_uids[family_map[uid]].append(uid)

    test_uids = set()
    for family in ordered_families:
        uids = family_to_uids.get(family, [])
        if not uids:
            continue
        test_count = 1 if len(uids) <= 3 else 2
        test_uids.update(uids[:test_count])
    return test_uids


def main():
    rng = random.Random(SEED)

    df = pd.read_csv(AUGMENTED_CSV)
    n_before = len(df)

    with open(TEST_SPLIT_PATH, 'rb') as f:
        split = pickle.load(f)
    train_idx = set(split['train_idx'])
    test_idx = set(split['test_idx'])
    assert not (train_idx & test_idx)

    sqli_pool_mask = df['source'] == 'sqli_pool'
    sqli_pool = df[sqli_pool_mask].copy()
    sqli_pool['in_train'] = sqli_pool.index.isin(train_idx)

    grp_purity = sqli_pool.groupby('source_uid')['in_train'].nunique()
    if (grp_purity > 1).any():
        raise RuntimeError(f"source_uid groups split across train/test: {grp_purity[grp_purity > 1].index.tolist()}")

    present_uids = set(sqli_pool['source_uid'].unique())
    global_test_uids = set(sqli_pool[~sqli_pool['in_train']]['source_uid'].unique())
    family_test_uids = compute_family_test_uids()

    eligible_uids = sorted(present_uids - global_test_uids - family_test_uids,
                            key=lambda x: int(x.split('_')[-1]))
    eligible_rows = sqli_pool[sqli_pool['source_uid'].isin(eligible_uids)]
    train_sqli_pool_rows = sqli_pool[sqli_pool['in_train']]

    n_eligible = len(eligible_rows)
    n_target = round(n_eligible * NOISE_FRACTION)

    sampled = eligible_rows.sample(n=n_target, random_state=SEED)

    new_rows = []
    sample_report = []
    skipped_no_keyword = 0
    for i, (_, row) in enumerate(sampled.iterrows()):
        content = str(row['content'])
        noised = noise_content(content, rng)
        if noised is None:
            skipped_no_keyword += 1
            continue
        orig_uid = row['source_uid']
        new_row = row.drop(labels=['in_train']).to_dict()
        new_row['content'] = noised
        new_row['source'] = 'sqli_pool_noise'
        new_row['source_uid'] = f'sqli_pool_noise_{i}'
        new_rows.append(new_row)
        if len(sample_report) < 10:
            sample_report.append((orig_uid, content, new_row['source_uid'], noised))

    # Append-only write: never re-serialize the existing 30084 rows, since a
    # read->concat->rewrite round trip re-escapes already-escaped backslash
    # content (confirmed via diff against a backup: 29 XSS payload rows with
    # literal '\' got double-escaped). Only the freshly generated new rows
    # are written, in append mode, using the file's existing column order.
    new_df = pd.DataFrame(new_rows).reindex(columns=df.columns)
    with open(AUGMENTED_CSV, 'a', newline='') as f:
        new_df.to_csv(f, index=False, header=False, quoting=csv.QUOTE_ALL, escapechar='\\')

    n_after = n_before + len(new_df)
    train_sqli_pool_after = len(train_sqli_pool_rows) + len(new_rows)
    noise_ratio_of_eligible = len(new_rows) / n_eligible if n_eligible else 0.0
    noise_ratio_of_original_train_sqli = len(new_rows) / len(train_sqli_pool_rows) if len(train_sqli_pool_rows) else 0.0
    noise_ratio_of_final_train_sqli = len(new_rows) / train_sqli_pool_after if train_sqli_pool_after else 0.0

    lines = []
    lines.append("=== Train-only SQLi comment-splitting-style noise augmentation ===")
    lines.append(f"Mechanism: random-position whitespace insertion inside a randomly chosen SQL keyword occurrence")
    lines.append(f"(DIFFERENT from held-out's fixed /**/ mid-keyword insertion in scripts/build_heldout_matrix_eval.py)")
    lines.append(f"Seed: {SEED}, target fraction of eligible rows: {NOISE_FRACTION}")
    lines.append("")
    lines.append(f"dataset rows before: {n_before}")
    lines.append(f"dataset rows after:  {n_after}")
    lines.append("")
    lines.append(f"sqli_pool unique source_uid present in dataset: {len(present_uids)}")
    lines.append(f"  excluded (global test_split_indices.pkl test_idx): {sorted(global_test_uids, key=lambda x:int(x.split('_')[-1]))}")
    lines.append(f"  excluded (SQLi family-test uids, per-family rule): {sorted(family_test_uids & present_uids, key=lambda x:int(x.split('_')[-1]))}")
    lines.append(f"  eligible for noise: {len(eligible_uids)} uids -> {eligible_uids}")
    lines.append("")
    lines.append(f"eligible rows (train, not touching either test split): {n_eligible}")
    lines.append(f"sampled for noising: {n_target} (skipped, no keyword match: {skipped_no_keyword})")
    lines.append(f"new noised rows appended: {len(new_rows)}")
    lines.append("")
    lines.append(f"total train sqli_pool rows BEFORE augmentation: {len(train_sqli_pool_rows)}")
    lines.append(f"total train sqli_pool rows AFTER augmentation (orig + noised): {train_sqli_pool_after}")
    lines.append(f"  noise ratio / eligible pool: {noise_ratio_of_eligible:.4f}")
    lines.append(f"  noise ratio / original train sqli_pool count: {noise_ratio_of_original_train_sqli:.4f}")
    lines.append(f"  noise ratio / final train sqli_pool count: {noise_ratio_of_final_train_sqli:.4f}")
    lines.append("")
    lines.append("=== Sample before/after (first 10) ===")
    for orig_uid, orig_content, new_uid, noised in sample_report:
        lines.append(f"{orig_uid} -> {new_uid}")
        lines.append(f"  before: {orig_content}")
        lines.append(f"  after:  {noised}")
    report = "\n".join(lines)
    REPORT_PATH.write_text(report + "\n")
    print(report)


if __name__ == '__main__':
    main()
