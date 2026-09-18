"""Prepare raw-text train/test CSVs for the RoBERTa/CodeBERT baselines,
using the EXACT same split as GATv2's `best_web_gnn_seed42.pth` (for a fair
comparison) -- no new split is computed here.

`data/test_split_indices.pkl`'s `train_idx` field is stale (frozen before
the three append-only augmentation rounds -- see
`src/training/train.py::_load_global_split()`'s docstring). The real split
GATv2 was trained/evaluated on is: `test_idx` exactly as frozen, `train_idx`
= every graph index NOT in `test_idx`, computed dynamically over the
CURRENT `web_graphs.pkl` (43659 rows). This script reproduces that logic
exactly rather than reading `train_idx` from the pickle.

Does not assume `augmented_web_attack.csv` row order matches
`web_graphs.pkl` graph order (even though `build_web_graphs()` does
preserve it by construction) -- for every single train/test index, this
script positionally pairs `df.iloc[i]` with `graphs[i]` and asserts their
`source_uid` AND `attack_type`/`y` agree before using the row, aborting on
any mismatch. (`source_uid` alone is not a unique key -- 18496 unique
values over 43659 rows, since multiple rows share a template/payload
source -- so this is a positional-alignment check backed by content
identity, not a `source_uid`-only lookup.)

Writes data/transformer_baseline/{train,test}.csv (columns: content,
attack_type).
"""
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
AUGMENTED_CSV = DATA_DIR / 'augmented_web_attack.csv'
GRAPHS_PKL = DATA_DIR / 'web_graphs.pkl'
SPLIT_PKL = DATA_DIR / 'test_split_indices.pkl'
OUT_DIR = DATA_DIR / 'transformer_baseline'

EXPECTED_TEST_BINCOUNT = [1550, 1214, 1451]
EXPECTED_TRAIN_BINCOUNT = [13450, 17381, 8613]


def main():
    df = pd.read_csv(AUGMENTED_CSV)
    with open(GRAPHS_PKL, 'rb') as f:
        graphs = pickle.load(f)['graphs']
    with open(SPLIT_PKL, 'rb') as f:
        split = pickle.load(f)

    assert len(df) == len(graphs), f"CSV rows ({len(df)}) != graph count ({len(graphs)})"

    test_idx = list(split['test_idx'])
    test_set = set(test_idx)
    train_idx = [i for i in range(len(graphs)) if i not in test_set]
    print(f"[*] train_idx = complement of frozen test_idx over range(len(graphs)), "
          f"NOT split['train_idx'] verbatim (see docstring) -- "
          f"split['train_idx'] has {len(split['train_idx'])} entries (stale), "
          f"computed train_idx has {len(train_idx)} entries")

    # positional-alignment verification: df.iloc[i] must be the same
    # underlying row as graphs[i] for every i in the split, not assumed.
    mismatches = []
    for i in list(train_idx) + list(test_idx):
        csv_uid = df.iloc[i]['source_uid']
        csv_label = int(df.iloc[i]['attack_type'])
        g = graphs[i]
        g_uid = g.source_uid
        g_label = int(g.y.item())
        if csv_uid != g_uid or csv_label != g_label:
            mismatches.append((i, csv_uid, g_uid, csv_label, g_label))
    if mismatches:
        print(f"[!] {len(mismatches)} positional mismatches between CSV and graphs, e.g.: {mismatches[:5]}")
        raise RuntimeError("CSV/graph row alignment verification FAILED -- stopping before writing any split.")
    print(f"[*] Verified source_uid + attack_type alignment for all {len(train_idx) + len(test_idx)} "
          f"train+test rows (CSV row i == graphs[i] for every i in the split).")

    def build_split(idx_list):
        rows = df.iloc[idx_list][['content', 'attack_type']].copy()
        rows['attack_type'] = rows['attack_type'].astype(int)
        return rows.reset_index(drop=True)

    train_df = build_split(train_idx)
    test_df = build_split(test_idx)

    train_bincount = [int((train_df['attack_type'] == c).sum()) for c in range(3)]
    test_bincount = [int((test_df['attack_type'] == c).sum()) for c in range(3)]

    print(f"[*] n_train={len(train_df)} bincount_train={train_bincount}")
    print(f"[*] n_test={len(test_df)} bincount_test={test_bincount}")

    if test_bincount != EXPECTED_TEST_BINCOUNT:
        raise RuntimeError(
            f"test bincount {test_bincount} != GATv2's documented "
            f"{EXPECTED_TEST_BINCOUNT} (docs/REPRODUCIBILITY.md) -- mapping bug, stopping.")
    if train_bincount != EXPECTED_TRAIN_BINCOUNT:
        raise RuntimeError(
            f"train bincount {train_bincount} != expected "
            f"{EXPECTED_TRAIN_BINCOUNT} (derived from docs/DATASET.md class totals minus the "
            f"frozen test bincount) -- mapping bug, stopping.")
    print("[*] bincounts match GATv2's documented split exactly -- safe to proceed.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUT_DIR / 'train.csv', index=False)
    test_df.to_csv(OUT_DIR / 'test.csv', index=False)
    print(f"[*] Wrote {OUT_DIR / 'train.csv'} and {OUT_DIR / 'test.csv'}")


if __name__ == '__main__':
    main()
