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

`scripts/add_xss_context_augmentation.py` (added later, same session) carries
the identical guard from the start (`source == 'xss_pool_context'`), applying
this pattern to every append-only augmentation script going forward.

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
**these are single-seed runs, not averaged over repeated seeds** (see
"5-seed statistics" in EXPERIMENT_LOG for the one exception, and its caveats).

**Same seed does NOT currently guarantee a bit-identical rerun on this
codebase/hardware — verified empirically, not just theoretically.**
Three independent `seed=42` training runs on identical code/config/data
(the originally-deployed checkpoint, the first iteration of
`scripts/run_5seed_stats.py`'s loop, and a standalone rerun) each produced a
**different** training trajectory (`epochs_run` 11 / 20 / 13; different
per-epoch loss/accuracy from epoch 2 onward — epoch 1 alone matched
bit-for-bit across all three, then diverged). Root cause: `torch`/CUDA
scatter-gather operations used by `GATv2Conv`/`global_max_pool` are not
deterministic by default on GPU.

**Attempted fix (2026-09-18, `src/utils/seed.py::set_seed()`):** added
`os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'` (required for
deterministic cuBLAS GEMM) and `torch.use_deterministic_algorithms(True)`
in a `try/except` (falls back with a printed `[!]` warning, not a silent
crash, if some op lacks a deterministic kernel on this torch/torch_geometric
version — no such fallback was triggered in the test below, i.e. every op
used claims to have a deterministic implementation available).

**Verified result: the fix did NOT eliminate the nondeterminism.** Two
`seed=42` runs after the fix (`epochs_run=13` both times, `best_epoch=3`
both times, epoch 1 bit-identical: `Loss: 0.0159 | Acc: 0.9993` both) still
diverge starting at epoch 2 (`Loss: 0.0013/Acc: 0.9995` vs.
`Loss: 0.0014/Acc: 0.9993`), and the resulting checkpoints differ
(`8a4cb7a0...` vs. `23f78313...`, md5). **This is the same divergence point
and a similar magnitude of divergence as the PRE-fix runs** (re-checked
against the saved pre-fix log: epoch 1 was `Loss: 0.0159 | Acc: 0.9993`
there too, epoch 2 was `Loss: 0.0015 | Acc: 0.9993` — a third distinct
value, different from either post-fix run) — i.e. `torch.use_deterministic_algorithms`
+ `CUBLAS_WORKSPACE_CONFIG` measurably changed nothing here. The fix is
still worth keeping (it's the documented, standard first step, fails loudly
via the warning instead of silently if it ever does matter, and costs
~40-60% more wall-clock time per epoch, which is the expected trade-off for
attempting determinism), but **do not claim it fixed reproducibility** — it
did not, on this codebase/torch/torch_geometric version combination.

**Tested and ruled out (2026-09-18):** the hypothesis that
`src/training/train.py`'s `DataLoader(train_data, batch_size=batch_size,
shuffle=True)` passing no explicit `generator=` was the (or a) cause —
`RandomSampler` without an explicit generator draws from the global
`torch.default_generator`, which *is* seeded by `set_seed()`'s
`torch.manual_seed(seed)`, so this was always a weaker hypothesis than it
looked; tested anyway since it's a classic, easy-to-miss PyTorch gotcha.
Added `generator=torch.Generator().manual_seed(seed)` to the `train_loader`
construction and ran two more `seed=42` runs on identical code/config/data.
**Still diverges from epoch 2**: epoch 1 matched bit-for-bit (`Loss: 0.0172
| Acc: 0.9988` both runs), epoch 2 didn't (`Loss: 0.0016/Acc: 0.9991` vs.
`Loss: 0.0015/Acc: 0.9993`), and `epochs_run` differed more than the
pre-fix baseline (13 vs. 15, `best_epoch` 3 vs. 5). The `generator=` change
is kept in `train.py` anyway — it's strictly more correct (an explicit,
independent RNG stream for the loader instead of implicitly sharing the
global default generator with every other RNG consumer in the training
loop) and doesn't cost anything — but **it does not fix reproducibility**,
consistent with the fact that it was never actually decoupled from
`torch.manual_seed()` in the first place.

**Remaining plausible contributor (hypothesis, not confirmed — out of scope
to chase further here):** the `Dropout(0.5)` in `src/models/layers.py`'s
classifier head draws from the CUDA RNG per-call during `model.train()`,
so its per-epoch draw sequence depends on exactly how many other CUDA RNG
draws happened earlier in that epoch — if that count itself varies run to
run (e.g. via the non-deterministic scatter-gather ops already identified
above), dropout's draws would desync and could plausibly explain divergence
starting at epoch 2 rather than epoch 1. Not tested in isolation; fixing
this would need its own verification pass, not assumed from this one.

**Closed (2026-09-18):** residual non-determinism (~epoch 2 onward, std
across 5 seeds ≈ 0.0001 on macro-F1) persists after ruling out
`torch.use_deterministic_algorithms` and explicit `DataLoader` generators;
likely attributable to CUDA-backed dropout sampling. Given the negligible
magnitude relative to all reported effects in this work, further isolation
was not pursued — seed=42 is documented as the reference run, with 5-seed
variance reported wherever a claim's sensitivity to it matters
(`results/final_stats_5seed.csv`).

**Practical implication for the paper:** report metrics as **mean ± std over
multiple seeds** (`results/final_stats_5seed.csv`), not as a single seed=42
number presented as exactly reproducible — the 5-seed spread already
measured there is a mix of genuine inter-seed variance and this
intra-seed-same-config nondeterminism, and the two cannot be cleanly
separated with the evidence collected so far.

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
| 5 | `best_web_gnn_seed42_PRE_xss_context.pth` | `logs/training_history_PRE_xss_context.log` | balanced class-weighted `CrossEntropyLoss` (`compute_class_weight('balanced', ...)`); no data change (§8) |
| 6 (current, deployed) | `best_web_gnn_seed42.pth` | `logs/training_history.log` | +64 train-only XSS context-distance (benign attribute padding) rows (`scripts/add_xss_context_augmentation.py`); see EXPERIMENT_LOG "XSS Context-Distance Augmentation" |

Final training set: 43,659 rows total (25,869 original + 2111 + 6400 + 5000 +
64), test set unchanged at 4215. `epochs_run=11`, seed=42 for every retrain
above except #4 (benign-syntax retrain, `epochs_run=17`, best_epoch=7 — the
only retrain where validation didn't converge at epoch 1; val_acc=0.9995 at
epoch 1 instead of 1.0) and #6 (`epochs_run=19`, best_epoch=9).

## 6-configuration ablation (edge types × edge_attr)

Branches off what is now `best_web_gnn_seed42_PRE_xss_context.pth` (retrain
#5 in the table above; it was still "current, deployed" and 43,595 rows at
the time this ablation ran, before the XSS context-distance augmentation
round added the 64 rows that produced retrain #6) — same 43,595-row train
set, same class-weighted loss. Full results/discussion: EXPERIMENT_LOG
§10-11, `results/ablation_edge_attr_seq_skip_sem.csv`.

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

All scripts below must be run as `python -m scripts.<name>` (no `.py`) from the
repo root — several import from `src/`, and `python scripts/<name>.py` fails
with `ModuleNotFoundError: No module named 'src'` (Python only puts the
script's own directory on `sys.path`, not the repo root). Verified: running
`python scripts/evaluate.py` directly does fail this way; `python -m
scripts.evaluate` does not.

```bash
source .venv/bin/activate

# 1. Label + balance/augment raw CSIC -> data/augmented_web_attack.csv (30,084 rows)
python -m scripts.prepare_data

# 2. Append-only train augmentation, in this order (each checked byte-identical on
#    the rows it doesn't touch, never touching test_split_indices.pkl):
python -m scripts.add_train_noise_augmentation       # +2111 rows (comment-split noise)
python -m scripts.add_missing_sqli_payloads_train     # +6400 rows (16 missing SQLi payloads)
python -m scripts.add_benign_syntax_diversity_train   # +5000 rows (benign header/JSON syntax)
python -m scripts.add_xss_context_augmentation        # +64 rows (XSS benign-attribute context padding)

# 3. Build the BAG graphs (sequential + skip + semantic edges; semantic_edges()
#    already includes the comment-split-survival fix)
python -m scripts.build_graphs

# 4. Train (seed=42 from configs/config.yaml, class-weighted loss)
python -m scripts.train_logwat

# 5. Evaluate on the frozen test split
python -m scripts.evaluate

# 6. Evaluate on the 90-sample held-out matrix (unseen evasion techniques)
python -m scripts.build_heldout_matrix_eval

# 6b. E_sem window-boundary control cells (in-range vs out-of-range token gap
#     for the XSS context-padding mechanism -- see EXPERIMENT_LOG)
python -m scripts.build_heldout_window_test

# 7. 6-configuration ablation (edge types x edge_attr)
python -m scripts.run_ablation_edge_attr_seq_skip_sem   # configs 1-4
python -m scripts.run_ablation_extra_no_edge_attr       # configs 5-6

# Optional: graph-size-only shortcut baseline (Decision Tree on [num_nodes, num_edges])
python -m scripts.decision_tree_size_baseline

# Optional: external evaluation dataset (Reviewer #1, evaluation-only, see
# docs/DATASET.md "External Evaluation Dataset")
python -m scripts.download_external_dataset
python -m scripts.prepare_external_dataset
python -m scripts.build_external_graphs
python -m scripts.evaluate_external_dataset

# Optional: 5-seed mean+-std statistics for the official config (see
# EXPERIMENT_LOG "5-seed statistics: official config"). Takes ~7-8 minutes
# on the GPU environment this was measured on. Back up
# data/models_pretrained/best_web_gnn_seed42.pth, logs/training_history.log,
# results/main_results.csv, and results/heldout_matrix_full.csv FIRST -- this
# script trains 5 more times and does NOT restore the deployed checkpoint
# itself (see its docstring); restore your backups afterward.
python -m scripts.run_5seed_stats

# Optional: per-request latency breakdown (CPU + GPU if available), see
# docs/LATENCY.md. Uses whatever is currently at
# data/models_pretrained/best_web_gnn_seed42.pth -- run against the restored
# deployed checkpoint, not mid-way through the 5-seed script above.
python -m scripts.benchmark_latency
```

To only re-evaluate the currently deployed checkpoint (no retraining, no dataset
changes), run steps 5-6 alone.
