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

## Transformer Baselines (Reviewer #3, 2026-09-18)

Real fine-tunes (not stubs) of `roberta-base` and `microsoft/codebert-base`
for 3-class web-attack classification on raw `content` text, added to
address Reviewer #3's request for a Transformer/LLM baseline comparison
(accuracy, latency, interpretability). Full narrative and findings:
`docs/EXPERIMENT_LOG_semantic_edge_investigation.md`'s Reviewer-#3-dated
sections (content duplication, held-out matrix comparison, external-dataset
analysis, interpretability). This section covers setup/reproducibility only.

**Environment:** same venv as the rest of this document
(`hin_web_vulne/web_venv`), plus `transformers==5.17.0`,
`tokenizers==0.23.2`, `safetensors==0.8.0`, `huggingface_hub==1.32.0`
(resolve cleanly against the pinned `torch==2.6.0+cu124`), and
`matplotlib==3.10.9` (interpretability plots only). Added to
`requirements.txt`.

**Data:** `data/transformer_baseline/{train,test}.csv` (gitignored,
regenerable), built by `scripts/prepare_transformer_baseline_data.py` from
the exact same frozen split GATv2 uses (`test_idx` from
`test_split_indices.pkl` as-is, `train_idx` = every other current graph
index, matching `src/training/train.py::_load_global_split()` — NOT
`test_split_indices.pkl`'s own stale `train_idx` field). Verified via
positional `source_uid`+`attack_type` alignment against
`data/web_graphs.pkl`, not assumed. Resulting bincounts (`train=[13450,
17381, 8613]`, `test=[1550, 1214, 1451]`) match this document's documented
GATv2 split exactly.

**Hyperparameters (both models, identical):** `batch_size=32`, `lr=2e-5`,
`max_length=128` (95th-percentile token length over train, per-tokenizer,
computed not guessed: p95=126, rounded up to the nearest multiple of 8),
up to 5 epochs with early stopping (`patience=2`, `min_delta=0`) on a
validation loss computed from a **5% stratified split carved out of TRAIN
only** (`val_fraction=0.05`, `random_state=seed`) — deliberately NOT
test-set accuracy, unlike GATv2's own training loop (a known methodology
weakness of this project, see the "Seed"/nondeterminism sections above;
these baselines avoid repeating it). `seed=42` via the existing
`src/utils/seed.py::set_seed()`, same `generator=`-seeded `DataLoader` as
`train.py`.

**Commands:**
```bash
source .venv/bin/activate  # or hin_web_vulne/web_venv, see top of this doc
python -m scripts.prepare_transformer_baseline_data
python -m scripts.finetune_roberta      # ~24 min on this GPU
python -m scripts.finetune_codebert     # ~30 min on this GPU
python -m scripts.evaluate_transformer_baselines   # test split + held-out + external
python -m scripts.evaluate_string_matching_test_split   # fills string-matching's one missing mode
python -m scripts.benchmark_latency_transformers   # ~5-8 min, same methodology as #11
python -m scripts.interpretability_attention_rollout
python -m scripts.build_final_baseline_comparison  # assembles the 4-method table from the above
```

**Fine-tune results (both converged the same way — best epoch 1, early-stopped
at epoch 3):**

| model | epochs_run | best_epoch | best_val_loss | wall time |
|---|---|---|---|---|
| RoBERTa | 3 | 1 (val_acc 0.9995) | 0.0054 | 1465s (~24.4 min) |
| CodeBERT | 3 | 1 (val_acc 0.9995) | 0.0051 | 1782s (~29.7 min; epoch 1 ran concurrently with a RoBERTa eval job on the same GPU, inflating its time — epochs 2-3 alone: 493s/635s) |

Checkpoints: `data/models_pretrained/{roberta,codebert}_baseline_seed42/`
(gitignored, ~500MB each). Logs: `logs/{roberta,codebert}_training.log`.

**Known limitation carried into every result reported for these baselines
(and for GATv2, and for string-matching — not specific to the new
baselines):** 25.2% of the frozen test split shares byte-identical
`content` with some train row (finite SQLi/XSS payload pools reused across
`source_uid` groups — not the previously-fixed `GroupShuffleSplit` bug, see
EXPERIMENT_LOG). Verified NOT silently inflating these numbers: both
baselines score ~1.0 on the duplicate-content AND the novel-content test
subsets alike (`results/transformer_baselines.csv`,
`test_split_duplication_breakdown` rows).

### TrafficLLM (`cui2025trafficllm`, cited in the paper's Related Work) — not run

Reviewer #3 also raised TrafficLLM specifically. Checked feasibility before
deciding, not skipped without reason: TrafficLLM
(arXiv:2504.04222, github.com/ZGC-LLM-Safety/TrafficLLM) is built on a
**ChatGLM2-6B backbone (6B parameters)**. Per the paper's own reported
figures, **training a new PEFT adaptation requires 23GB GPU memory** (14h,
20k steps on 50k samples), and **inference alone requires 13GB**. This
machine's GPU has **8GB total VRAM** (7.2GB free at idle) — insufficient
for TrafficLLM even for inference-only use, let alone fine-tuning, by a
wide margin (13GB inference requirement vs. 8GB total capacity). Not run.
This is a hardware constraint, not a scope decision: RoBERTa/CodeBERT
(125M/125M params) were feasible on this GPU precisely because they are
roughly 50x smaller than TrafficLLM's backbone.

## Graph & Sequence Baselines — baseline-fairness re-run (#16 / RQF-06, 2026-09-19)

The paper's original Table 2 (TextCNN / Bi-LSTM / Stack-LSTM / GCN / GraphSAGE / GIN /
GATv2 / HGT / Proposed) was measured on the original 30k-row dataset/split, with baseline
configurations for which no code or hyperparameters survive in this repo. #16 re-runs
seven of those baselines on the **current** dataset (43,659 rows), the **frozen** split
(`test_split_indices.pkl`), and the **same training loop and protocol as GATv2**
(`src/training/train.py::train()`, parameterized rather than copied). Narrative, tables and
findings: `docs/EXPERIMENT_LOG_semantic_edge_investigation.md`, section "Baseline fairness
re-run (#16)". This section is setup/reproducibility only. ("GATv2 (vanilla)" from the old
Table 2 was **not** re-run: "vanilla" has no definition in this repo.)

**Scope:** both groups, 7 baselines, seed 42 for the canonical tables plus seeds 42–46 for
all of them (and for GATv2, through the same evaluation code, as the reference).

**Group A — graph** (`src/models/baselines_graph.py`): each is `GATBackbone`'s skeleton — 2
conv layers at `hidden_dim=256`, BatchNorm, ELU, `JumpingKnowledge('cat')`, and the identical
`Classifier` head (global max-pool → MLP) — with **only the conv operator swapped**, on the
same graphs (same node features, same E_seq+E_skip+E_sem `edge_index`, same split).

| model | conv | params | edge types seen |
|---|---|---|---|
| GATv2 (reference) | `GATv2Conv`, 8 heads × 32 | 299,011 | none (edge_index merged) |
| GCN | `GCNConv` | 215,555 | none |
| GraphSAGE | `SAGEConv` (mean aggr) | 297,475 | none |
| GIN | `GINConv`, 2-layer MLP update net, `train_eps=True` | 347,141 | none |
| HGT | `HGTConv`, 8 heads, 1 node type × 3 edge types (seq/skip/sem) | 610,357 | **yes** — from the `edge_attr` one-hot |

HGT needs relation types, so it trains on `data/web_graphs_edge_attr.pkl` (same graphs +
`edge_attr`, built by `scripts/build_edge_attr_graphs.py`, which **asserts** per graph that
`x`/`edge_index`/`y`/`source_uid` are identical to `web_graphs.pkl` for all 43,659 train
graphs; same for `external_dataset_graphs_edge_attr.pkl`). `web_graphs.pkl` itself is never
overwritten. GCN/GraphSAGE/GIN ignore `edge_attr` by construction.

**Group B — sequence** (`src/models/baselines_seq.py`): input is the raw token sequence
`web_security_tokenizer(content)`, embedding (dim 128) **learned from scratch** (no pretrained
vectors — like GATv2, unlike RoBERTa/CodeBERT). Vocabulary = tokens with frequency ≥ 2 in the
**train** rows only (11,918 tokens + PAD/UNK = 11,920); everything else is UNK (test-split UNK
rate 6.8%). Sequences are stored as `Data` objects (`x` = token ids, empty `edge_index`) in
`data/sequence_baseline/web_sequences.pkl`, so they run through the same DataLoader/train loop
(`to_dense_batch` re-pads them). `prepare_sequence_baseline_data.py` asserts label and
per-row node-count parity with `web_graphs.pkl` for all rows. Max sequence length is 106
tokens; nothing is truncated.

| model | architecture | params |
|---|---|---|
| Bi-LSTM | embedding → 1-layer BiLSTM (128/dir) → [masked mean ; masked max] pool → MLP head | 1,922,051 |
| TextCNN (Kim 2014) | embedding → Conv1d k=3,4,5 × 100 filters → max-over-time → dropout 0.5 → linear | 1,680,563 |
| StackLSTM | embedding → 2 stacked unidirectional LSTM layers (256) → last hidden state → MLP head | 2,513,923 |

Note the parameter counts: ~1.5M of each sequence model's parameters is the embedding table,
so they are 6–8× larger than GATv2. They also see none of the hand-crafted per-token features
(danger-char / SQL-keyword flags, entropy) that the graph models' node features contain.

**Training protocol — identical for all 7, and to GATv2** (`configs/config.yaml`): AdamW
`lr=5e-4`, `weight_decay=0.1`, `batch_size=64`, class-weighted CrossEntropy (balanced weights
from train), `ReduceLROnPlateau(mode='max', factor=0.5, patience=5)`, max 100 epochs, early
stopping `patience=10`/`min_delta=0` on test-split accuracy, `seed` via `set_seed()` (including
its `use_deterministic_algorithms(True)`), `DataLoader` generator seeded. **No baseline needed
or received a learning-rate override** — every run converged at the default LR (best test acc
≥ 0.9972 within the first ≤ 11 epochs) and every run stopped by early stopping (11–21 epochs),
none reaching the 100-epoch cap. **Inherited methodology weakness:** this loop selects the best
checkpoint and stops on *test-split* accuracy (unlike the Transformer baselines, which use a
validation loss from a train-only split). The same protocol applies to every graph/sequence
baseline and GATv2, so the comparison is like-for-like, but the absolute test-split numbers
are selected on the test split.

**Commands** (venv `hin_web_vulne/web_venv`; run from repo root):
```bash
python -m scripts.build_edge_attr_graphs            # HGT's graphs (train + external), ~2 min, verifies parity
python -m scripts.prepare_sequence_baseline_data    # vocab + token-id sequences, verifies parity
python -m scripts.train_baseline --model gcn --seed 42        # also: graphsage gin hgt bilstm textcnn stacklstm
python -m scripts.evaluate_baselines --group graph            # -> results/graph_baselines.csv    (seed 42)
python -m scripts.evaluate_baselines --group sequence         # -> results/sequence_baselines.csv (seed 42)
python -m scripts.evaluate_baselines --group graph --seeds 42 43 44 45 46     # + results/graph_baselines_5seed.csv
python -m scripts.evaluate_baselines --group sequence --seeds 42 43 44 45 46  # + results/sequence_baselines_5seed.csv
python -m scripts.evaluate_baselines --group reference --seeds 42 43 44 45 46 # GATv2 -> results/gatv2_reference_5seed.csv
python -m scripts.benchmark_latency_graph_baselines           # -> results/latency_breakdown_graph_baselines.csv (GPU idle!)
python -m scripts.build_final_baseline_comparison             # 11 methods x 3 modes
python -m scripts.build_table2_old_vs_new                     # -> results/table2_old_vs_new.csv
```
Outputs: checkpoints `data/models_pretrained/baseline_<model>_seed<seed>.pth`; training logs
`logs/<model>_training.log` (all seeds, one session header each; `logs/` and `data/*.pkl` are
gitignored, regenerable); results CSVs in the same long schema as `transformer_baselines.csv`.
`evaluate_baselines` backs up any existing output CSV to `_PRE_<timestamp>.csv` and only
replaces the rows of the methods it re-evaluates. The GATv2 reference reuses the deployed
`best_web_gnn_seed42.pth` and the 5-seed run's `best_web_gnn_seed43..46.pth` (never retrained).

**Evaluation pipeline check:** the same `predict`/held-out/external code on the *deployed GATv2
checkpoint* reproduces the previously committed numbers exactly (external weighted F1
0.839892 / macro F1 0.631240, held-out 80/90, test 5-seed mean 0.99981±0.00011, per-seed held-out
cells 8/7/8/8 for seeds 43–46) — verified, not assumed.

**Training runs, seed 42** (epochs_run / best_epoch / wall time; wall times for seeds 43–46 are
inflated because 3 training streams shared the GPU, so they are not comparable to these):

| model | epochs_run | best_epoch | best test acc | wall time |
|---|---|---|---|---|
| GCN | 18 | 8 | 1.0000 | 182 s |
| GraphSAGE | 12 | 2 | 0.9998 | 93 s |
| GIN | 15 | 5 | 0.9998 | 111 s |
| HGT | 16 | 6 | 1.0000 | 701 s |
| Bi-LSTM | 13 | 3 | 1.0000 | 140 s |
| TextCNN | 14 | 4 | 1.0000 | 195 s |
| StackLSTM | 16 | 6 | 1.0000 | 291 s |

**Latency** (Group A only; `docs/LATENCY.md` methodology, GATv2 measured in the same run): the
script deliberately does **not** call `set_seed()` — a first attempt that did (deterministic
algorithms on) inflated GPU forward times ~2× (GATv2 CUDA 4.0 ms vs 2.18 ms). Run with the GPU
idle. Group B latency was not measured (not required; architecture too different to compare).

**Code change to shared training code:** `src/training/train.py::train()` gained optional
`model_factory`, `model_save_path`, `log_path`, `data_path`, `run_tag`, `write_family_split`
arguments and now returns a run-summary dict. Defaults reproduce the deployed GATv2 run (the
diff substitutes those names for the module constants, wraps the SQLi family-split step in
`if write_family_split:` and adds the return value); baselines pass
`write_family_split=False` so a baseline run never touches `sqli_family_test_indices.pkl`.
