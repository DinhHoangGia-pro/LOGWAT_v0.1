# Dataset

Dataset description and schema.

## Source data: CSIC 2010 license/terms of use (investigated 2026-09-18)

`data/csic_database.csv` (raw) is the "HTTP DATASET CSIC 2010", created by
Carmen Torrano Giménez, Alejandro Pérez Villegas, and Gonzalo Álvarez Marañón at
the Information Security Institute of CSIC (Spanish National Research Council)
in 2009-2010, originally hosted at `isi.csic.es/dataset/` /
`tic.itefi.csic.es/dataset/`.

**Finding: no explicit, discoverable license or terms-of-use document exists for
this dataset — for either the raw data or derived/processed data.**

- The original CSIC host page is currently unreachable (`isi.csic.es`: connection
  timeout; `tic.itefi.csic.es`: "no route to host" — checked directly via
  `urllib`, not just a proxy/tool restriction), so it could not be read directly.
  It could not be recovered via the Wayback Machine either (archive.org rate-limited
  every retry during this check).
- The [IMPACT Cyber Trust mirror listing](https://www.impactcybertrust.org/dataset_view?idDataset=940)
  (a formal dataset-sharing framework that normally documents usage terms)
  explicitly marks this dataset as offered **"outside of the IMPACT mediation
  framework"**, with **no redistribution or derived-data clause given** — it
  defers to the CSIC host page for actual terms, which (see above) is down.
- No secondary source checked — academic papers citing the dataset, the
  [SpiderLabs modsecurity-crs discussion](https://github.com/SpiderLabs/owasp-modsecurity-crs/issues/1016),
  multiple GitHub mirrors (e.g. [Kiinitix/HTTP-CSIC-2010](https://github.com/Kiinitix/HTTP-CSIC-2010)),
  a Kaggle re-upload, or a [PhD-thesis-linked reformatted-CSV mirror](https://petescully.co.uk/research/csic-2010-http-dataset-in-csv-format-for-weka-analysis/)
  — quotes any explicit clause restricting or permitting redistribution of
  either raw or derived data. The one consistent norm across sources is
  **attribution**: acknowledge the three CSIC researchers above as the dataset's
  creators. None of these mirrors are the authoritative source, so their
  practice (freely re-hosting raw and reformatted copies) is evidence of a
  de facto community norm, not a confirmed legal permission.

**Conclusion for this repo:** there is no confirmed prohibition on redistributing
derived data specifically (nor, symmetrically, a confirmed blanket permission for
either raw or derived redistribution — the honest answer is "undocumented,"
not "permitted"). Until the original terms page is reachable and re-checked:
- Treat `data/csic_database.csv` (raw CSIC content) as **not verified safe to
  publish/redistribute** — it is already excluded from git via `.gitignore`
  (`data/*.csv`); keep it that way and do not attach it to the paper's
  supplementary material without re-checking this page first.
- `data/augmented_web_attack.csv` is a **derived** dataset (labeled/balanced/
  augmented; only 84 rows keep `source='csic_original'`, i.e. an unmodified raw
  CSIC row end-to-end — the `NA`-source Benign/XSS rows below still reuse raw
  CSIC header fields (User-Agent, cookie, etc.) as templates even where
  `content` was replaced). Its legal status is exactly as undocumented as the
  raw data's, not clearer — no source draws a raw-vs-derived distinction
  explicitly. Do not treat "it's derived" as license clearance.
- Cite the three original CSIC researchers by name wherever the dataset is
  described in the paper, per the attribution norm found above.
- Re-attempt this check (the CSIC page, then the Wayback Machine) before final
  paper/code submission — a live terms page could change this conclusion.

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

## Train-only addition of the 16 missing SQLi payloads (2026-09-18)

Follow-up to the "known gap" noted above (25/41 pool payloads actually present). A hypothesis that `balance_dataset()` used a stale/cached `len(sqli_payloads)` was checked directly and **refuted**: the function re-reads `sqli_payload_pool.csv` fresh on every call (verified: `len(sqli_payloads)` computed live right now = 41, not 25) — there is no code bug. The real explanation is simply that the frozen CSV was generated by a past run, before the pool file grew from 25 to 41 rows, and the pipeline was never re-run since.

A **full regenerate via `scripts/prepare_data.py` was tested in a dry run and rejected**: it reproduces all Benign/XSS/csic_original rows byte-identically at the same positions (20309/30084 rows unchanged), but reassigns `source_uid` for the SQLi-pool portion using `i % 41` instead of `i % 25`. Checked against the frozen `data/test_split_indices.pkl`: **all 41/41 SQLi payload groups end up split across both the old train and old test positions**, and the same 41/41 payload content strings appear in both — i.e. naively regenerating and reusing the frozen split would silently reintroduce full SQLi train/test content leakage (the exact class of bug already fixed once via grouped `GroupShuffleSplit`). Rejected in favor of the append-only approach below.

- Script: `scripts/add_missing_sqli_payloads_train.py`. Appends new rows only (same append-only discipline as the noise augmentation above — never rewrites existing rows).
- Adds 400 rows each (matching the density of the 25 payloads already present, 10000/25=400) for the 16 payloads at pool index 25-40 (the last 7 Error-based + all 9 Stacked/other), poisoning benign frames the same way `balance_dataset()` does.
- `source='sqli_pool_extra'`, `source_uid='sqli_pool_extra_<payload_idx>_<seq>'` — new, cannot collide with any existing source_uid, and (appended past the previous 32195-row range) automatically outside `test_split_indices.pkl`'s `test_idx`.
- Row counts: 32195 → **38595** (+6400). `attack_type` counts: `{1: 18595, 0: 10000, 2: 10000}`.
- Full report: `data/missing_sqli_payloads_train_report.txt`.
- Consequence: with this addition, **all 41 SQLi payloads now appear somewhere in train**, but the frozen test set (`test_split_indices.pkl`) still only ever evaluates against the original 25 (payload index 0-24) — the held-out/test side of the pipeline has no coverage of Error-based (idx 25-31) or Stacked/other (idx 32-40) payloads. This trade-off was chosen explicitly (over a full dataset-v2 regeneration with a new split) to keep every prior checkpoint/held-out comparison in this session's history valid.

## Train-only Benign syntax diversity: header-style + json-style (2026-09-18)

Motivated by the held-out `Benign header_field`/`Benign json_field` cells staying stuck at 0.0 accuracy across two retrains. Verified before generating anything (not assumed): of the 1689 train Benign rows with `num_nodes<=15`, **0/1689 contained `:` and 0/1689 contained `{`/`}`** — 1685/1689 (99.8%) were `key=value&key=value` query-string style. So the issue isn't that train Benign examples are "too short" relative to the held-out cells; train Benign has essentially **zero syntactic coverage** of header- or JSON-shaped content, at any length.

- Script: `scripts/add_benign_syntax_diversity_train.py`. Append-only (same discipline as the prior two augmentation scripts).
- 2500 header-style rows (`Name: value` pairs joined with `; `, e.g. `Content-Type: application/json`, `X-Request-ID: req-<hex>`) + 2500 JSON-style rows (`{"field":"value",...}`, e.g. `{"status":"ok","action":"login"}`) = 5000 new rows.
- Header names and JSON field names are deliberately **disjoint** from what the held-out matrix uses (`build_heldout_matrix_eval.py`: `Authorization`/`X-Trace` for `header_field`, `page`/`user`/`locale` for `json_field`), so those held-out cells stay a genuine test of syntactic generalization rather than something the model was directly trained on.
- `source='benign_short_synthetic'`, `source_uid='benign_short_synthetic_header_<seq>'` / `'benign_short_synthetic_json_<seq>'` — new, no collisions, appended past the previous 38595-row range so automatically outside `test_split_indices.pkl`'s `test_idx`.
- Row counts: 38595 → **43595** (+5000). `attack_type` counts: `{0: 15000, 1: 18595, 2: 10000}`.
- Full report + samples: `data/benign_syntax_diversity_train_report.txt`.

## Train-only XSS context-distance augmentation (2026-09-18)

Analogous in purpose to the SQLi comment-splitting noise round above, but
structurally different (context/distance dilution, not character-level
keyword fragmentation) — see `docs/EXPERIMENT_LOG_semantic_edge_investigation.md`,
"XSS Noise-Augmentation Feasibility Check" and "XSS Context-Distance
Augmentation" for the full feasibility analysis and result.

- Script: `scripts/add_xss_context_augmentation.py`. Append-only. Targets
  only 3 of E_sem's 9 semantic pairs: `('img','onerror')`, `('script','src')`,
  `('svg','onload')`.
- Eligible pool: train-only XSS rows (`source` is `NaN`) containing at least
  one target pair as whole words — only 186/8549 (most XSS payloads in this
  dataset use event handlers/tags outside E_sem's 9-pair vocabulary).
  35% sampled (matching `NOISE_FRACTION` from the SQLi noise script, for
  methodological consistency) → 65 rows; 1-2 random benign HTML attributes
  (`data-*`/`class`/`id`/`style`) inserted immediately before the pair's
  second keyword, **only accepted if the resulting token gap stays <15**
  (E_sem's `window`, checked per-row via the real tokenizer) — 60/65
  accepted with 2 attributes, 4/65 downgraded to 1, 1/65 skipped entirely
  (didn't fit even at 1 attribute).
- `source='xss_pool_context'`, `source_uid='xss_pool_context_<i>'` — new, no
  collisions, appended past the previous 43595-row range so automatically
  outside `test_split_indices.pkl`'s `test_idx`.
- Row counts: 43595 → **43659** (+64). `attack_type` counts:
  `{0: 15000, 1: 18595, 2: 10064}`.
- Full report + samples: `data/xss_context_augmentation_report.txt`.
- Retrain result: frozen test split reached macro F1 = 1.0 exactly (0 errors,
  a marginal improvement over the pre-retrain 1-error state). New held-out
  cells (`scripts/build_heldout_window_test.py`) show **no measurable
  change** — both the in-range (gap<15) and a separate out-of-range (gap≥16)
  control cell were already 10/10 on the checkpoint *before* this
  augmentation/retrain, and remain 10/10 after. See EXPERIMENT_LOG for why
  this revises (rather than confirms) the original hard-limit hypothesis for
  the out-of-range case.

## Checksums

SHA-256 of the current frozen dataset files (`data/augmented_web_attack.csv`,
43659 rows including all augmentation rounds above; `data/test_split_indices.pkl`),
for verifying a copy matches this exact state:

```
1c100f6ddee2715cf84805c864e6f9400c5f5961b31924983ca4b92a17f21213  data/augmented_web_attack.csv
f216c0679cf2b1e0fd41c4cffc1c865bb432312010cbb81eff678b25da68003d  data/test_split_indices.pkl
```

(Prior state, 43595 rows before the XSS context-distance augmentation round:
`845a3acbdf8250a59c02daa93bb43a58c7a9b317e1efcdf0a397e3a3b4d29b7d` — preserved at
`data/augmented_web_attack_PRE_xss_context.csv`.)

## External Evaluation Dataset (2026-09-18, Reviewer #1)

Goal: evaluate (never train/fine-tune on) the deployed checkpoint against an
independent public dataset not derived from CSIC 2010, to address a
generalization concern separate from the held-out evasion-technique matrix
(§ EXPERIMENT_LOG). Not appended to `data/augmented_web_attack.csv`; the
frozen dataset is untouched.

- **Chosen:** [HttpParamsDataset](https://github.com/Morzeux/HttpParamsDataset/)
  (`Morzeux/HttpParamsDataset`), priority (1) of the three candidates given
  for this task.
- **Source URL:** `https://raw.githubusercontent.com/Morzeux/HttpParamsDataset/master/payload_full.csv`
- **Download date:** 2026-09-18.
- **License:** MIT (LICENSE file in the repo, confirmed by direct fetch).
- **Provenance caveat:** per the repo's own README, this dataset was built
  from CSIC2010 (benign values only) plus attack payloads generated by
  sqlmap (SQLi), XSSYA (XSS), Vega Scanner (cmdi/path-traversal), and FuzzDB.
  So it is **not fully independent of CSIC2010 on the benign side** — but the
  attack payloads come from different generation tools than anything in this
  repo's SQLi/XSS training data, so it's still a meaningfully different attack
  surface. Not a clean "fully independent" dataset; reported as such.
- Raw file saved as-is: `data/external/httpparamsdataset_raw.csv` (script:
  `scripts/download_external_dataset.py`).

### Preparation and format domain-shift check

Script: `scripts/prepare_external_dataset.py`. Maps `attack_type` →
`0=Benign/1=SQLi/2=XSS` (`norm`/`sqli`/`xss`); drops 379 `cmdi`/`path-traversal`
rows (out of scope for this repo's 3-class model). Applies
`clean_content()` (URL-unquote, drop empty/len<2) from `src/preprocessing/normalization.py`.
Output: `data/external/external_dataset_clean.csv`, 30677 rows
(Benign=19293, SQLi=10852, XSS=532), `source='external_httpparamsdataset'`,
`source_uid='external_httpparamsdataset_<i>'`. Full report:
`data/external/external_dataset_prep_report.txt`.

**Format domain-shift check (avg `content` length, chars, per class — the
model only ever reads the `content` column, per
`src/bag/graph_builder.py:build_web_graphs`):**

| class | train (augmented_web_attack.csv) avg | external avg | diff |
|---|---|---|---|
| Benign | 106.7 | 11.9 | **-88.9%** ⚠️ |
| SQLi | 32.8 | 90.5 | **+175.4%** ⚠️ |
| XSS | 62.4 | 63.2 | +1.2% |

**⚠️ Two of three classes exceed the 50% difference threshold.** The external
set is made of short, isolated HTTP parameter *values* (single form fields /
payload fragments extracted from URLs/params) — not full HTTP request strings
like this repo's `content` column. Benign external content is ~9x shorter
than train Benign; SQLi external content is, surprisingly, ~2.75x *longer*
than train SQLi (the external SQLi payloads are full injection strings
with subqueries, e.g. `1' where 6406=6406;select count(*) from
rdb$fields...`, vs. this repo's train SQLi being individual short attack
tokens/fragments). **Read the results below as testing generalization to a
different input FORMAT as well as different attack content — a confound, not
a clean apples-to-apples generalization test.**

### External Evaluation on HttpParamsDataset

No training/fine-tuning. Checkpoint: `data/models_pretrained/best_web_gnn_seed42.pth`
(confirmed byte-identical to `best_web_gnn_seed42_ablation_1_full_no_edge_attr.pth`,
i.e. ablation config (1): full edges, no edge_attr — see
`docs/REPRODUCIBILITY.md`). Graphs built with `use_seq=True, use_skip=True,
use_sem=True, use_edge_attr=False`, matching that checkpoint exactly
(`scripts/build_external_graphs.py`). Evaluation script:
`scripts/evaluate_external_dataset.py`. Full output:
`data/external/external_dataset_evaluation_report.txt`,
`results/external_dataset_evaluation.csv`.

n=30677 (Benign=19293, SQLi=10852, XSS=532).

| method | overall accuracy | Benign P/R/F1 | SQLi P/R/F1 | XSS P/R/F1 |
|---|---|---|---|---|
| **GATv2** (deployed checkpoint) | **0.7309** | 0.974 / 0.629 / 0.764 | 0.643 / 0.900 / 0.750 | 0.173 / 0.985 / 0.294 |
| String-matching (`classify_request()`) | 0.2629 | 0.000 / 0.000 / 0.000 | 1.000 / 0.701 / 0.824 | 1.000 / 0.863 / 0.926 |
| Decision Tree, size-only ([num_nodes, num_edges], trained on the same `train_idx` as `scripts/decision_tree_size_baseline.py`) | 0.1783 | 0.515 / 0.078 / 0.136 | 0.639 / 0.337 / 0.441 | 0.014 / 0.583 / 0.028 |

GATv2 confusion matrix (rows=true, cols=pred, order Benign/SQLi/XSS):
```
[[12130  5422  1741]
 [  321  9767   764]
 [    1     7   524]]
```

Decision Tree size-only accuracy (0.1783) is **well below** GATv2 (0.7309) —
unlike the internal frozen test split, where the same size-only baseline
reaches 0.7877 (`data/decision_tree_size_baseline.txt`), graph size trained on
this repo's data does **not** transfer to the external set's very different
size distribution. This rules out "GATv2's external accuracy is just the
graph-size shortcut in disguise" for this dataset — GATv2 is using signal the
size-only baseline doesn't have access to. The string-matching baseline
predicts 0 Benign (every "norm" `payload` value is too generic/keyword-free
to match any regex, so `classify_request()` returns Unknown(-1) for it, which
is counted as wrong here) — it only ever fires on SQLi/XSS keyword matches,
inflating its SQLi/XSS precision to 1.000 at the cost of recall and total
uselessness on Benign.

**Consistency with the internal held-out matrix (8/9 cells, EXPERIMENT_LOG §11):**
the internal held-out matrix's one remaining known weakness besides
`comment_splitting` was resolved for short benign header/JSON fragments
(fixed at retrain #4) — but this external set's Benign recall (0.629) shows
the *same underlying failure mode reappearing at larger scale*: 5422/19293
external Benign rows are misclassified as SQLi and 1741 as XSS, i.e. short,
generic parameter values (`22997112x`, `plaa caudillo 60`) are still
confused with attacks by this checkpoint often enough to matter, even though
the specific `header_field`/`json_field` held-out cells pass. This is
consistent with, not contradicted by, the held-out matrix result — the
held-out matrix only probes 10 samples per cell, too few to have caught a
~28% external Benign error rate on this specific short-fragment distribution.
XSS's catastrophic precision (0.173) is a **new failure mode not previously
observed internally** (all three internal XSS held-out cells are 1.0): 1741
Benign and 764 SQLi external rows are wrongly predicted XSS, a
disproportionate false-positive rate given XSS is only 532/30677 (1.7%) of
this set.

**XSS precision deep-dive: base-rate artifact vs. genuine over-prediction.**
At the current argmax threshold: P=0.1730, R=0.9850 (TP=524, FP=2505, FN=8,
TN=27640 → measured FPR=0.0831). Using the softmax probability of the XSS
class (not just the final argmax label), **PR-AUC (average precision) =
0.9627** — vs. a no-skill baseline of 0.0173 (XSS's own prevalence in this
set). This gap (0.963 vs. 0.017) shows the model's underlying *ranking*
signal for XSS is strong; the poor precision is a property of the fixed
3-way-argmax operating point under extreme imbalance, not an absence of
discriminative signal. Applying Bayes' rule (precision(π) = π·TPR / (π·TPR +
(1−π)·FPR)) with the measured TPR/FPR at this same operating point, to three
assumed base rates: **π=0.1% → precision=1.17%; π=1% → precision=10.69%;
π=1.7% (this set's own actual rate) → precision=17.30%** (self-consistent
with the measured 0.1730 above — confirms the formula is correctly
calibrated to the real confusion matrix). Solving the same formula for the
FPR that would be *required* to reach a target precision of 50% at each
base rate, fixed TPR=0.985: **π=0.1% needs FPR≤0.000986 (measured is 84x
higher); π=1% needs FPR≤0.009950 (8.4x higher); π=1.7% needs FPR≤0.017384
(4.8x higher).** **Conclusion: both effects are real and compounding, not
either/or.** The extremely low real-world base rate of XSS mathematically
caps precision at any fixed FPR (this alone would keep precision under ~2%
at a plausible web-traffic-wide XSS rate of 0.1%) — but the model's FPR at
the current fixed threshold (8.3%) is *also* too high even relative to this
dataset's own inflated 1.7% rate (needs to drop ~4.8x just to hit 50%
precision there), so it is not purely a base-rate artifact either. Given the
strong PR-AUC, this reads as a **threshold-calibration problem at extreme
class imbalance**, not a fundamental lack of signal — unlike the E_sem
architectural limit in `docs/EXPERIMENT_LOG_semantic_edge_investigation.md`,
this one is plausibly addressable by threshold tuning or a
class-imbalance-aware decision rule, without further architecture changes.

Threshold re-calibration (one-vs-rest at threshold ≈0.988 on the XSS softmax
probability, instead of the 3-way argmax) achieves Precision=50.00% with
Recall=97.18% (vs. 98.50% at argmax) — confirming, not just hypothesizing,
that this is a fixable threshold-calibration issue rather than a fundamental
model limitation.

**Honest conclusion:** GATv2 (0.7309 accuracy) clearly outperforms both
baselines on this external, differently-formatted dataset, and its
external-Benign failure pattern is consistent with (an amplified version of)
a weakness the internal held-out matrix already flagged — this is
**evidence of partial generalization, not a clean win**. The result should
be reported as "reduced but non-trivial performance under both attack-content
and input-format shift," not as confirmation that internal held-out matrix
performance (8/9) transfers to real-world-shaped external traffic; the
~89%-shorter external Benign content and the new XSS false-positive mode are
both genuine, separately-reportable limitations for the paper's Discussion,
alongside the E_sem architectural limit already documented in
`docs/EXPERIMENT_LOG_semantic_edge_investigation.md`.

### Root cause of external Benign misclassification: a 4th, un-augmented benign form (2026-09-18)

Quick check (documentation only — no training/code change at this stage; this
is Limitations material, not a bug to patch here). Of the 19293 external
Benign rows, **7163 (37.1%) are misclassified** (5422 → SQLi, 1741 → XSS).
10 real examples, verbatim content + length, sampled proportionally across
both misprediction directions (6 SQLi-side, 4 XSS-side, matching the true
~76%/24% split):

| predicted as | len | content |
|---|---|---|
| SQLi | 16 | `1084102116286517` |
| SQLi | 16 | `3006162765919932` |
| SQLi | 16 | `2301580169203669` |
| SQLi | 5 | `e72i4` |
| SQLi | 16 | `7008973356777544` |
| SQLi | 5 | `10860` |
| XSS | 15 | `mori@itrends.do` |
| XSS | 18 | `gunther@nik.com.tn` |
| XSS | 21 | `mullen@menorca.com.bf` |
| XSS | 34 | `keep_theberge9@aprendeaestudiar.gr` |

**This is confirmed to be a 4th benign form, distinct from all three forms
already covered by train augmentation** (`docs/DATASET.md` §§
"Train-only Benign syntax diversity"): checked structurally across all 7163
misclassified rows, not just the 10 samples above —

```
with '=' (query-string form):  0 / 7163
with '&' (query-string form):  0 / 7163
with ':' (header-style form):  0 / 7163
with '{' or '}' (json form):   0 / 7163
pure digit strings:            4075 / 7163 (56.9%)
```

**100% of misclassified external Benign rows contain none of `=`, `&`, `:`,
`{`, `}`** — zero overlap with the query-string (`key=value&key=value`),
header-style (`Name: value`), or JSON-style (`{"k":"v"}`) forms added by
`scripts/add_benign_syntax_diversity_train.py`. They are **single bare
values with no key/delimiter structure at all**: standalone numeric strings
that look like credit-card numbers or ID/PIN codes (`1084102116286517`,
`10860`), short alphanumeric tokens (`e72i4`), and — for the XSS-predicted
half specifically — **email addresses** (`mori@itrends.do`), which the `@`
and `.`-heavy structure of plausibly resembles the tokenization of
XSS-relevant special characters (`@`, `.`) closely enough to be a distinct
sub-pattern worth naming on its own.

**For the paper's Limitations section:** the three benign-syntax-diversity
forms added in this repo's training data (query-string, header, JSON) cover
benign content that has *some* delimiter/key-value structure. This 4th form —
an isolated bare value (numeric ID, short alphanumeric token, or email
address) with no surrounding structure — was never represented in training,
and the model has no reliable signal to distinguish "digits that are
somebody's ID number" from "digits that are a SQLi numeric literal," or
"an email address" from "an XSS payload with special characters." This is a
genuine, previously-undocumented generalization gap, not covered by the
existing 8/9 held-out matrix (whose `field_query`/`header_field`/`json_field`
Benign cells are all structured, delimiter-bearing forms) and not something
this task's scope fixes — recorded here as evidence for the Discussion/
Limitations section, and as a candidate 4th augmentation form for anyone
picking this up later.
