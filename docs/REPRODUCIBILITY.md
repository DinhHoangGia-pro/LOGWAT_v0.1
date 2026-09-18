# Reproducibility

## ⚠️ Data Integrity Guards

The three append-only train-augmentation scripts
(`scripts/add_train_noise_augmentation.py`, `scripts/add_missing_sqli_payloads_train.py`,
`scripts/add_benign_syntax_diversity_train.py` — see the command sequence below)
originally had **no protection against being run twice**. Each generates its new
rows' `source_uid` from a counter (`seq`/`i`) that restarts at 0 on every call, so a
second run wouldn't just duplicate rows — it would emit `source_uid` values that
collide with the previous run's, breaking the global `source_uid` uniqueness that
`GroupShuffleSplit` and the SQLi family-test split logic depend on. This isn't just
extra rows; it's a corrupted train/test split. Confirmed empirically via an isolated
dry-run against a scratch copy of the dataset (real file never touched, verified by
md5sum before/after).

Fixed in commit `088603c`: each script now raises `RuntimeError` at the top of
`main()` if its `source` value is already present in
`data/augmented_web_attack.csv`.

**If you intentionally want to regenerate one of these augmentations** (e.g. to
build a new frozen dataset version), you must manually delete the rows with the
matching `source` value first — do not just remove or comment out the guard, since
that reintroduces the exact `source_uid` collision risk above.

## Environment

All numbers in `results/` and `docs/EXPERIMENT_LOG_semantic_edge_investigation.md`
were produced with the venv at `hin_web_vulne/web_venv` (Python 3.10.20, CUDA 12.4).
Exact package pins: [`requirements.txt`](../requirements.txt) (root). Setup:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install torch_scatter==2.1.2+pt26cu124 torch_sparse==0.6.18+pt26cu124 \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

## Seed

Global seed = **42**, set in `configs/config.yaml` (`seed: 42`) and applied via
`src/utils/seed.py::set_seed()` at the top of `src/training/train.py::train()`.
The same seed was used for every retrain and every ablation configuration below —
**these are single-seed runs, not averaged over repeated seeds.** The one place
this matters most: the `case_mixing` regressions in ablation configs (3)-(6)
(`docs/EXPERIMENT_LOG_semantic_edge_investigation.md` §10-11) are flagged there as
a preliminary single-run signal, not a variance-checked causal claim.

## Frozen dataset and split

`data/augmented_web_attack.csv` and `data/test_split_indices.pkl` are frozen
(`dataset-v1-frozen` tag, commit `d475bc4`) — see `docs/DATASET.md`. The test split
is fixed at 4215 rows (bincount `[1550, 1214, 1451]`) for every retrain below; only
the train side grew, via three append-only augmentation scripts (never touching
`test_idx`). Do not regenerate `augmented_web_attack.csv` or re-run the three
augmentation scripts below on the current working tree — they are append-only and
their output is already baked into the frozen file; re-running them would duplicate
rows.

## Retrain history and checkpoints

Six retrains were run in sequence, each building on the previous one's data/code
state. Every checkpoint except the last was renamed to `..._PRE_<next-change>.pth`
before being overwritten, so all six are preserved in
`data/models_pretrained/`:

| # | checkpoint file | training log | what changed vs. previous retrain |
|---|---|---|---|
| 0 (pre-investigation baseline) | `best_web_gnn_seed42_PRE_semantic_fix.pth` | `logs/training_history_PRE_semantic_fix.log` | starting point of the investigation (§1 of EXPERIMENT_LOG) |
| 1 | `best_web_gnn_seed42_PRE_valfix_and_noise.pth` | `logs/training_history_PRE_valfix_and_noise.log` | `semantic_edges()` fix (`_reconstruct_keywords()`, survives comment-splitting) + graph rebuild; same data (§3-4) |
| 2 | `best_web_gnn_seed42_PRE_missing_payloads.pth` | `logs/training_history_PRE_missing_payloads.log` | fixed `train.py` validation-split bug (was validating against an SQLi-only split) + 2111 train-only comment-split-noise rows (`scripts/add_train_noise_augmentation.py`) (§5) |
| 3 | `best_web_gnn_seed42_PRE_benign_syntax.pth` | `logs/training_history_PRE_benign_syntax.log` | +6400 train-only rows for the 16 SQLi payloads missing from the frozen dataset (`scripts/add_missing_sqli_payloads_train.py`) (§6) |
| 4 | `best_web_gnn_seed42_PRE_class_weight.pth` | `logs/training_history_PRE_class_weight.log` | +5000 train-only benign header/JSON syntax rows (`scripts/add_benign_syntax_diversity_train.py`) (§7) |
| 5 (current, deployed) | `best_web_gnn_seed42.pth` | `logs/training_history.log` | balanced class-weighted `CrossEntropyLoss` (`compute_class_weight('balanced', ...)`); no data change (§8) |

Final training set: 43,595 rows total (25,869 original + 2111 + 6400 + 5000), test
set unchanged at 4215. `epochs_run=11`, seed=42 for every retrain above except #4
(benign-syntax retrain, `epochs_run=17`, best_epoch=7 — the only retrain where
validation didn't converge at epoch 1; val_acc=0.9995 at epoch 1 instead of 1.0).

## 6-configuration ablation (edge types × edge_attr)

Branches off retrain #5's dataset/checkpoint (same 43,595-row train set, same
class-weighted loss). Full results/discussion: EXPERIMENT_LOG §10-11,
`results/ablation_edge_attr_seq_skip_sem.csv`.

| config | checkpoint | E_seq | E_skip | E_sem | edge_attr | driver script |
|---|---|---|---|---|---|---|
| (1) full, no edge_attr | `best_web_gnn_seed42_ablation_1_full_no_edge_attr.pth` | ✓ | ✓ | ✓ |  | reuses retrain #5, re-evaluated only |
| (2) full + edge_attr | `best_web_gnn_seed42_ablation_2_full_edge_attr.pth` | ✓ | ✓ | ✓ | ✓ | `scripts/run_ablation_edge_attr_seq_skip_sem.py` |
| (3) no-skip + edge_attr | `best_web_gnn_seed42_ablation_3_no_skip_edge_attr.pth` | ✓ |  | ✓ | ✓ | `scripts/run_ablation_edge_attr_seq_skip_sem.py` |
| (4) no-sem + edge_attr | `best_web_gnn_seed42_ablation_4_no_sem_edge_attr.pth` | ✓ | ✓ |  | ✓ | `scripts/run_ablation_edge_attr_seq_skip_sem.py` |
| (5) no-skip, no edge_attr | `best_web_gnn_seed42_ablation_5_no_skip_no_edge_attr.pth` | ✓ |  | ✓ |  | `scripts/run_ablation_extra_no_edge_attr.py` |
| (6) no-sem, no edge_attr | `best_web_gnn_seed42_ablation_6_no_sem_no_edge_attr.pth` | ✓ | ✓ |  |  | `scripts/run_ablation_extra_no_edge_attr.py` |

Both driver scripts rebuild `data/web_graphs.pkl` with the matching
`use_seq`/`use_skip`/`use_sem`/`use_edge_attr` toggles per config, retrain from
scratch (config (1) excepted, reused as-is), evaluate on both the frozen test split
and the 90-sample held-out matrix, and **restore `best_web_gnn_seed42.pth`,
`logs/training_history.log`, `results/main_results.csv`, and
`results/heldout_matrix_full.csv` to the retrain-#5 state afterward** (verified by
checkpoint md5sum) — running them does not permanently change the deployed model.

## Full command sequence

The frozen dataset already includes the output of the three augmentation scripts
below — **do not re-run them against the current `data/augmented_web_attack.csv`.**
This sequence documents how it was produced and is meant for building an equivalent
dataset from scratch (e.g. a fresh clone before the freeze, or a deliberate
new frozen version decided with the user):

```bash
source .venv/bin/activate

# 1. Label + balance/augment raw CSIC -> data/augmented_web_attack.csv (30,084 rows)
python scripts/prepare_data.py

# 2. Append-only train augmentation, in this order (each checked byte-identical on
#    the rows it doesn't touch, never touching test_split_indices.pkl):
python scripts/add_train_noise_augmentation.py       # +2111 rows (comment-split noise)
python scripts/add_missing_sqli_payloads_train.py     # +6400 rows (16 missing SQLi payloads)
python scripts/add_benign_syntax_diversity_train.py   # +5000 rows (benign header/JSON syntax)

# 3. Build the BAG graphs (sequential + skip + semantic edges; semantic_edges()
#    already includes the comment-split-survival fix)
python scripts/build_graphs.py

# 4. Train (seed=42 from configs/config.yaml, class-weighted loss)
python scripts/train_logwat.py

# 5. Evaluate on the frozen test split
python scripts/evaluate.py

# 6. Evaluate on the 90-sample held-out matrix (unseen evasion techniques)
python scripts/build_heldout_matrix_eval.py

# 7. 6-configuration ablation (edge types x edge_attr)
python scripts/run_ablation_edge_attr_seq_skip_sem.py   # configs 1-4
python scripts/run_ablation_extra_no_edge_attr.py       # configs 5-6

# Optional: graph-size-only shortcut baseline (Decision Tree on [num_nodes, num_edges])
python scripts/decision_tree_size_baseline.py
```

To only re-evaluate the currently deployed checkpoint (no retraining, no dataset
changes), run steps 5-6 alone.
