"""Add train-only XSS rows testing E_sem robustness to CONTEXT-DISTANCE noise
(benign HTML attribute insertion between a semantic-pair's two keywords),
structurally different from `add_train_noise_augmentation.py`'s SQLi
comment-splitting noise (character-level keyword fragmentation). See
`docs/EXPERIMENT_LOG_semantic_edge_investigation.md`, "XSS Noise-Augmentation
Feasibility Check", for the feasibility analysis this implements.

Only 3 of E_sem's 9 semantic pairs are targeted: ('img','onerror'),
('script','src'), ('svg','onload'). NOT ('script','eval')/('onerror','eval')
-- those pairs' second keyword ('eval') sits inside a JS expression /
attribute VALUE, not the HTML tag's attribute-name-list, so inserting
HTML-attribute-style text there would not be a valid "benign attribute
insertion" the way it is for the other 3 (see EXPERIMENT_LOG for the full
reasoning).

Mechanism, deliberately IN-RANGE (gap always kept <15, i.e. inside E_sem's
`window`) -- this augments the case E_sem CAN still bridge, to test whether
training on realistic attribute-padded variants improves robustness within
the window. The gap>=15 case is a separate, proven architectural limit (see
EXPERIMENT_LOG "E_sem hard distance threshold") and is NOT what this script
targets -- no amount of data can fix that case, so none is added for it.

For each selected row: insert 1-2 random benign HTML attributes
(data-*/class/id/style, random values) immediately before the first
occurrence of the pair's second keyword. Verified via the real
`web_security_tokenizer()` + locating both keyword token indices that the
resulting gap is <15 BEFORE accepting; retries with fewer attributes, skips
the row entirely if even 1 attribute doesn't fit under the window.

Two independently-seeded random streams (matching the SQLi noise script's
convention): stream A (pandas, random_state=SEED) samples which eligible
rows to augment; stream B (`random.Random(SEED)`) decides each row's number/
type/value of inserted attributes.
"""
import csv
import pickle
import random
import re
import string
from pathlib import Path

import pandas as pd

from src.preprocessing.tokenizer import web_security_tokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
AUGMENTED_CSV = DATA_DIR / 'augmented_web_attack.csv'
TEST_SPLIT_PATH = DATA_DIR / 'test_split_indices.pkl'
REPORT_PATH = DATA_DIR / 'xss_context_augmentation_report.txt'

NOISE_FRACTION = 0.35  # matches add_train_noise_augmentation.py, for methodological consistency
SEED = 42
WINDOW = 15  # must match src/edges/semantic.py::semantic_edges()'s default window

TARGET_PAIRS = [('img', 'onerror'), ('script', 'src'), ('svg', 'onload')]
_PAIR_RE = {
    (k1, k2): (re.compile(r'\b' + k1 + r'\b', re.IGNORECASE), re.compile(r'\b' + k2 + r'\b', re.IGNORECASE))
    for k1, k2 in TARGET_PAIRS
}

_RAND_CHARS = string.ascii_lowercase + string.digits


def _rand_token(rng, n=6):
    return ''.join(rng.choice(_RAND_CHARS) for _ in range(n))


def make_attr(rng):
    """One random, semantically-inert HTML attribute: name=value."""
    kind = rng.choice(['data', 'class', 'id', 'style'])
    if kind == 'data':
        return f'data-{_rand_token(rng, 4)}={_rand_token(rng, 6)}'
    if kind == 'class':
        return f'class={_rand_token(rng, 6)}'
    if kind == 'id':
        return f'id={_rand_token(rng, 6)}'
    # style: a simple, plausible-looking CSS declaration
    prop = rng.choice(['color', 'display', 'opacity', 'width'])
    val = rng.choice(['none', 'red', '0', '10px', '#' + _rand_token(rng, 6)])
    return f'style={prop}:{val}'


def find_target_pair(content):
    """Return (kw1, kw2, m2) for the first TARGET_PAIRS entry both present
    in `content` (whole-word, case-insensitive), or None."""
    for (k1, k2), (r1, r2) in _PAIR_RE.items():
        m1 = r1.search(content)
        m2 = r2.search(content)
        if m1 and m2:
            return k1, k2, m2
    return None


def token_gap(content, kw1, kw2):
    """First-occurrence token-index gap between kw1 and kw2 in the tokenized
    content, or None if either keyword doesn't survive tokenization."""
    tokens = web_security_tokenizer(content)
    try:
        idx1 = tokens.index(kw1.lower())
        idx2 = tokens.index(kw2.lower())
    except ValueError:
        return None
    return idx2 - idx1


def try_augment(content, kw1, kw2, m2, rng, max_attrs=2):
    """Try inserting `max_attrs`, then fewer, benign attributes right before
    kw2's first occurrence, accepting the first variant whose token gap is
    <WINDOW. Returns (new_content, n_attrs_used) or (None, 0) if nothing fits."""
    for n_attrs in range(max_attrs, 0, -1):
        attrs = ' '.join(make_attr(rng) for _ in range(n_attrs))
        new_content = content[:m2.start()] + attrs + ' ' + content[m2.start():]
        gap = token_gap(new_content, kw1, kw2)
        if gap is not None and gap < WINDOW:
            return new_content, n_attrs
    return None, 0


def main():
    rng = random.Random(SEED)

    df = pd.read_csv(AUGMENTED_CSV)
    n_before = len(df)

    if df['source'].eq('xss_pool_context').any():
        raise RuntimeError(
            "Dữ liệu nguồn 'xss_pool_context' đã tồn tại trong augmented_web_attack.csv "
            "— dừng để tránh append trùng. Xoá thủ công nếu thực sự muốn chạy lại."
        )

    with open(TEST_SPLIT_PATH, 'rb') as f:
        split = pickle.load(f)
    train_idx = set(split['train_idx'])
    test_idx = set(split['test_idx'])
    assert not (train_idx & test_idx)

    xss_mask = (df['attack_type'] == 2) & df['source'].isna()
    xss = df[xss_mask].copy()
    xss_train = xss[xss.index.isin(train_idx)]

    matches = xss_train['content'].astype(str).apply(find_target_pair)
    eligible_mask = matches.notna()
    eligible_rows = xss_train[eligible_mask].copy()
    eligible_rows['_pair'] = matches[eligible_mask]

    pair_counts = {p: 0 for p in TARGET_PAIRS}
    for p in eligible_rows['_pair']:
        pair_counts[(p[0], p[1])] += 1

    n_eligible = len(eligible_rows)
    n_target = round(n_eligible * NOISE_FRACTION)
    sampled = eligible_rows.sample(n=n_target, random_state=SEED)  # stream A

    new_rows = []
    sample_report = []
    skipped_no_fit = 0
    attrs_used_counts = {1: 0, 2: 0}
    for i, (_, row) in enumerate(sampled.iterrows()):
        content = str(row['content'])
        kw1, kw2, m2 = row['_pair']
        new_content, n_attrs = try_augment(content, kw1, kw2, m2, rng, max_attrs=2)  # stream B
        if new_content is None:
            skipped_no_fit += 1
            continue
        attrs_used_counts[n_attrs] += 1
        new_row = row.drop(labels=['_pair']).to_dict()
        new_row['content'] = new_content
        new_row['source'] = 'xss_pool_context'
        new_row['source_uid'] = f'xss_pool_context_{i}'
        new_rows.append(new_row)
        if len(sample_report) < 10:
            gap = token_gap(new_content, kw1, kw2)
            sample_report.append((kw1, kw2, n_attrs, gap, content, new_content))

    new_df = pd.DataFrame(new_rows).reindex(columns=df.columns)
    with open(AUGMENTED_CSV, 'a', newline='') as f:
        new_df.to_csv(f, index=False, header=False, quoting=csv.QUOTE_ALL, escapechar='\\')

    n_after = n_before + len(new_df)

    lines = []
    lines.append("=== Train-only XSS context-distance (benign attribute padding) augmentation ===")
    lines.append("Mechanism: insert 1-2 random benign HTML attributes (data-*/class/id/style) "
                 "immediately before the pair's second keyword, keeping the token gap <15 "
                 "(E_sem's window) -- verified per-row via the real tokenizer before accepting.")
    lines.append(f"Seed: {SEED}, target fraction of eligible rows: {NOISE_FRACTION}")
    lines.append(f"Target pairs: {TARGET_PAIRS} (script/eval and onerror/eval excluded -- see script docstring)")
    lines.append("")
    lines.append(f"dataset rows before: {n_before}")
    lines.append(f"dataset rows after:  {n_after}")
    lines.append("")
    lines.append(f"XSS rows (source=NaN, mechanism-1 original): {len(xss)}")
    lines.append(f"  train-only (source_uid/index in train_idx): {len(xss_train)}")
    lines.append(f"  eligible (contains >=1 target pair as whole words): {n_eligible}")
    lines.append(f"    by pair: {pair_counts}")
    lines.append("")
    lines.append(f"sampled for augmentation: {n_target}")
    lines.append(f"  accepted with 2 attributes: {attrs_used_counts[2]}")
    lines.append(f"  accepted with 1 attribute (2 didn't fit under window=15): {attrs_used_counts[1]}")
    lines.append(f"  skipped (neither 1 nor 2 attributes kept gap <15): {skipped_no_fit}")
    lines.append(f"new rows appended: {len(new_rows)}")
    lines.append("")
    lines.append("=== Sample before/after (first 10) ===")
    for kw1, kw2, n_attrs, gap, before, after in sample_report:
        lines.append(f"pair={kw1}/{kw2}  n_attrs={n_attrs}  resulting_gap={gap}")
        lines.append(f"  before: {before}")
        lines.append(f"  after:  {after}")
    report = "\n".join(lines)
    REPORT_PATH.write_text(report + "\n")
    print(report)


if __name__ == '__main__':
    main()
