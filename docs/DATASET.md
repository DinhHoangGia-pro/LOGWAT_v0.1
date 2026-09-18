# Dataset

Dataset description and schema.

## Frozen dataset v1

- Freeze time: 2026-09-17 11:35:16 +0700
- Git commit: d475bc4f87c7ff26bdca8d0e51e21285c70df386
- Git tag: dataset-v1-frozen

### Dataset rows and class counts

- Total rows: 30084
- attack_type counts: {0: 10000, 1: 10084, 2: 10000}
- source counts: {'NA': 20000, 'sqli_pool': 10000, 'csic_original': 84}
- source_uid non-null rows: 30084

### SQLi payload pool counts per family

- Boolean-based: 8
- Error-based: 8
- Time-based: 8
- UNION-based: 8
- Stacked/other: 9

### Notes

This is the exact frozen dataset revision used for the full 100-epoch training run and evaluation. Any later reported metrics should be traced back to this commit/tag and the corresponding split artifact data/test_split_indices.pkl.

**Known gap (found 2026-09-18, not yet fixed):** `data/sqli_payload_pool.csv` currently has 41 payloads across 5 families (8/8/8/8/9), but the frozen `augmented_web_attack.csv` above only actually contains rows for the first 25 (`sqli_pool_0`..`sqli_pool_24` — 8 Boolean + 8 UNION + 8 Time + 1 Error-based). The remaining 7 Error-based and all 9 Stacked/other payloads were added to the pool CSV after the dataset freeze and were never incorporated. `sqli_payload_pool.csv` has no git history (untracked), so the exact pool state at freeze time cannot be recovered. The family-breakdown counts above describe the current pool file, not what the frozen dataset actually trains/tests on.

## Train-only comment-splitting noise augmentation (2026-09-18)

Added AFTER the v1 freeze, as an addition on top of the frozen rows (none of the original 30084 rows were modified — verified byte-identical against `data/augmented_web_attack_PRE_train_noise.csv`, a durable backup of the pre-augmentation file kept in this same `data/` directory, not in `/tmp`).

- Script: `scripts/add_train_noise_augmentation.py`. Appends new rows only; never rewrites existing rows (an earlier attempt that read+rewrote the whole file was found to double-escape backslash characters in 29 unrelated XSS payload rows and was reverted).
- Mechanism: insert 1-3 whitespace characters at a **random interior position** inside a **randomly chosen** SQL keyword occurrence (broader keyword list than E_sem's 9-pair vocabulary: select/union/from/where/order/group/insert/into/drop/table/delete/update/and/or/null/waitfor/delay/benchmark/sleep/extractvalue/updatexml/convert/cast/database/version/concat/exec/all/limit). This is deliberately a **different mechanism** from the held-out set's fixed `/**/` mid-keyword insertion (`scripts/build_heldout_matrix_eval.py`, cell `SQLi comment_splitting`), so that held-out cell remains a genuine test of generalization to an unseen evasion technique rather than something the model was directly trained on.
- Eligibility: only `source='sqli_pool'` rows whose `source_uid` is (a) in `data/test_split_indices.pkl`'s `train_idx` (never `test_idx`), AND (b) not one of the SQLi-family test uids selected by the same deterministic per-family rule used in `src/training/train.py::_fixed_family_group_split` (first 2 payload ids per family, sorted by index; 1 for families with ≤3 payloads). Of the 25 `sqli_pool` uids actually present in the dataset, 3 are excluded by (a) (`sqli_pool_6/7/16`) and 7 more by (b) (`sqli_pool_0/1/8/9/16/17/24`, `16` already excluded by (a)) — leaving 16 eligible uids / 6400 eligible rows.
- Sampling: seed=42, 35% of the 6400 eligible rows sampled (2240), 129 skipped (no keyword match found in that payload's content) → **2111 new rows appended**, `source='sqli_pool_noise'`, `source_uid='sqli_pool_noise_0'..'sqli_pool_noise_2110'` (all new, no collision with any existing source_uid).
- Row counts: dataset total 30084 → 32195. Train sqli_pool rows: 8800 (original, all kept unchanged) → 10911 after adding the 2111 noised siblings.
  - Noised rows as % of eligible pool: 2111/6400 = **32.98%**
  - Noised rows as % of original train sqli_pool count: 2111/8800 = **23.99%**
  - Noised rows as % of final train sqli_pool count: 2111/10911 = **19.35%** (i.e. train sqli_pool is now ~80.65% original-form / ~19.35% noised-form)
- Full report with before/after samples: `data/train_noise_augmentation_report.txt`.
- `data/test_split_indices.pkl` was not modified (confirmed: no new row index appears in either its `train_idx` or `test_idx`, since new rows were appended past index 30083). `data/sqli_family_test_indices.pkl` was likewise not touched by this step.
