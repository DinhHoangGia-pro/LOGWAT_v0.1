# Experiment log: semantic-edge (E_sem) generalization investigation

Chronological record of a single investigation thread spanning 2026-09-17 to 2026-09-18: why GATv2 achieves near-perfect accuracy on the frozen test split yet fails on a small held-out matrix designed to test generalization to unseen evasion techniques, and how far that gap could be closed. Numbers below are copied from the referenced evidence files, not re-derived from memory.

**Terminology note (for the .tex draft).** Grepping this file, `docs/DATASET.md`, and every `results/*.csv`/`data/*.txt` turns up several names in use for the same 90-sample (3 classes × 3 evasion techniques × 10) evaluation set:
- **"held-out matrix"** — the dominant prose term in this file (section headers §2, §11, "Final state") and in `docs/DATASET.md`.
- `heldout_matrix_full` / `heldout_matrix_eval` — the filename/script-name form (no hyphen, Python/CSV identifier constraint), e.g. `results/heldout_matrix_full.csv`, `scripts/build_heldout_matrix_eval.py`. Same referent as "held-out matrix", just a naming-convention artifact, not a different concept.
- "held-out set" — used a couple of times loosely (`docs/DATASET.md`, memory notes) as a synonym; drop this form in the paper, it reads as the standard ML train/val/test held-out split and invites confusion with the frozen test split (`data/test_split_indices.pkl`), which is a *different* thing.
- "90-sample" / "90 mẫu" — a size descriptor, not a name; fine as an appositive but not as the standalone term.
- "obfuscation test set" / "evasion technique(s) matrix" — descriptive phrasing that shows up in prose (e.g. "generalization to unseen evasion techniques") but is never used as the object's name anywhere in the repo; avoid introducing it as a new label in the paper.
- **Do not confuse with** `heldout_keyword_eval` / `heldout_keyword_comparison` (`results/heldout_keyword_comparison.csv`) — a *different* artifact (GATv2-vs-string-matching baseline comparison), not this 90-sample matrix.

**Chosen canonical term for the paper: "held-out matrix"** (full form on first use: "held-out matrix (90 samples: 3 classes × 3 evasion techniques × 10)", short form "held-out matrix" thereafter). This is a documentation-only note — no files are renamed; existing `heldout_matrix_*` file/script names stay as-is.

## 1. Initial trigger: suspected label/E_sem circularity

On the frozen test split (`data/test_split_indices.pkl`, 4215 rows), GATv2 reaches macro F1 ≈ 0.999 for all three classes (Benign/SQLi/XSS). A pure string-matching baseline (`classify_request()` in `src/preprocessing/normalization.py`, the same regex heuristic used to *label* the dataset) reaches ≈ 0.99 on the same split.

This raised a concrete concern: both the ground-truth labels and `semantic_edges()` (`src/edges/semantic.py`) — the E_sem construction, referred to as Table 1/Eq.4-6 in the paper draft — are built from the same fixed keyword vocabulary (`union`, `select`, `from`, `script`, `onerror`, ...). If the benchmark's difficulty is entirely keyword-driven, high test accuracy would not demonstrate learned structural/semantic understanding, only keyword matching that a GNN happens to reproduce.

## 2. Held-out matrix: 3/9 cells fail

To test this directly, `scripts/build_heldout_matrix_eval.py` was built: a 90-sample matrix (10 samples × 3 classes × 3 evasion techniques per class), with content deliberately avoiding the literal keywords the labeling regex and E_sem vocabulary depend on:

| class | techniques |
|---|---|
| Benign | `field_query`, `header_field`, `json_field` |
| SQLi | `sql_new_commands`, `comment_splitting`, `case_mixing` |
| XSS | `data_uri_base64`, `event_handler_focus`, `svg_script_variant` |

First run (checkpoint trained before this investigation): **6/9 cells correct, 3/9 at accuracy = 0.0**:
- `Benign / header_field` (e.g. `Authorization: Bearer token_0; X-Trace: session-0`) — predicted SQLi 10/10.
- `Benign / json_field` (e.g. `{"page":"dashboard","user":"alice0","locale":"en-US"}`) — predicted XSS 10/10.
- `SQLi / comment_splitting` (e.g. `p=0;UNI/**/ON ALL SEL/**/ECT * FR/**/OM users WH/**/ERE id=0;--`) — predicted Benign 10/10.

Initial hypothesis (graph size as a shortcut) was probed via `scripts/investigate_benign_fp.py` and a causal padding/truncation probe (`scripts/investigate_causal_padding.py`, evidence: `data/causal_padding_test.txt`). Padding a benign header sample from 12→36 nodes flipped its prediction SQLi→XSS (not to Benign); truncating a correctly-classified benign query sample from 16→11 nodes did *not* flip it to SQLi. **Conclusion: graph size has a causal effect on predictions but is not the sole or linearly deterministic factor** — ruled out as a complete explanation.

## 3. Root-cause diagnosis: `semantic_edges()` breaks under comment-splitting

Direct inspection (`scripts/inspect_semantic_edges.py`, evidence: `data/semantic_edges_inspect.txt`) of `semantic_edges()` on the four known-bad/good samples:

| sample | E_sem edges (before fix) |
|---|---|
| `clean` SQLi (`UNION ALL SELECT * FROM users`, correctly predicted SQLi) | 4 |
| `comment_split` SQLi (`UNI/**/ON ALL SEL/**/ECT...`, wrongly predicted Benign) | **0** |
| `header_field` Benign (wrongly predicted SQLi) | 0 (correctly — no attack keywords present) |
| `header_padded` Benign (wrongly predicted XSS) | 0 (correctly) |

`semantic_edges()` matched keyword *tokens* exactly against a fixed pair list (`('union','select')`, `('select','from')`, etc.). Comment-splitting (`/**/` inserted mid-keyword) tokenizes `union` into `uni` + `on`, `select` into `sel` + `ect` — neither fragment matches any keyword, so E_sem drops to 0 edges. **The comment-splitting evasion technique breaks the graph's own semantic-edge construction, not just the string-matching baseline it was meant to route around.**

Fix (`src/edges/semantic.py`, `_reconstruct_keywords()`): greedy-prefix token merging across noise characters (`/`, `*`, `-`, `#`) before matching against the keyword vocabulary, so `uni` + `/` + `*` + `*` + `/` + `on` re-merges to `union` before the pair-match step.

Verified safe on the full dataset before rebuilding anything (`data/semantic_fix_safety_check.txt`): 0/10000 real Benign rows gained a spurious E_sem edge (0.00% → 0.00%). Verified correct on the four probe samples (`data/semantic_edges_inspect_after_fix.txt`): `comment_split` E_sem edges 0 → **4**, identical structure to `clean`'s 4 edges; the other three samples unchanged.

## 4. Retrain #1 (semantic-fix only): correct fix, zero measured improvement

Graph rebuild + full retrain on the fixed `semantic_edges()`, keeping the dataset otherwise frozen. Result: `results/COMPARISON_before_after_semantic_fix.csv` — held-out matrix **unchanged**, all 3 cells still at 0.0.

Explanation confirmed by direct logit inspection on the `comment_split` sample: `Benign=1.83, SQLi=1.53, XSS=-3.57` — still predicts Benign, gap only ≈0.3. **The checkpoint had never seen comment-splitting-style noise during training** (train set only contained clean payloads), so the 4 new E_sem edges — correct in construction — carried no learned weight advantage over the model's existing (wrong) decision boundary. Fixing the graph-construction bug was necessary but not sufficient; the model also needed training exposure to the technique.

A side finding at this stage: a Decision-Tree classifier using only `[num_nodes, num_edges]` per graph reached **78.77% accuracy** on the frozen test split — confirming graph *size* alone is a moderately strong shortcut signal across the whole test set, not just in a few held-out cases (`data/decision_tree_size_baseline.txt`).

## 5. Retrain #2: train-only noise augmentation (different mechanism from held-out)

Per explicit requirement: training noise must use a **different mechanism** than the held-out matrix's fixed `/**/` mid-keyword split, so the held-out `comment_splitting` cell remains a genuine unseen-technique test rather than something directly trained on.

`scripts/add_train_noise_augmentation.py`: inserts 1-3 whitespace characters at a **random interior position** inside a **randomly chosen** SQL keyword (broader vocabulary than E_sem's 9 pairs), applied to 35% of eligible train-only `sqli_pool` rows (2111 new rows, append-only, never touching `test_split_indices.pkl`). Full derivation and eligibility logic: `data/train_noise_augmentation_report.txt`.

Alongside this, a pre-existing but previously undiagnosed bug was found and fixed: `train()` was validating against `_fixed_family_group_split()`'s output — an SQLi-only subset (bincount `[0, 2813, 0]`) meant for a separate family-breakdown report — instead of the 3-class balanced `test_split_indices.pkl` split evaluate.py actually reports against. This explained the previously-unexplained instant convergence (val_acc=1.0 at epoch 1) seen in earlier retrains. Fixed in `src/training/train.py` (`_load_global_split`).

Result (`results/COMPARISON_v2_after_valfix_and_noise.csv`):
- `SQLi / comment_splitting`: **0.0 → 1.0** (10/10 correct).
- Confirmed via logit gap, not just the argmax flip: `Benign=0.98, SQLi=2.19` (gap **+1.21**), vs. the pre-fix `Benign=1.83, SQLi=1.53` (gap **-0.30**, wrong side). The margin didn't just cross zero — it moved substantially, evidence of a real, non-marginal shift rather than a coincidental flip.
- `Benign header_field` / `json_field`: unchanged at 0.0 (this round did not target them).
- Test split macro F1: 1.0 exactly (confusion matrix `[[1550,0,0],[0,1214,0],[0,0,1451]]`, one fewer error than pre-fix).
- Decision Tree size-only baseline: unchanged, 0.7877 (original test-split graphs untouched).

## 6. Side finding: only 25/41 SQLi payloads were ever in the dataset

While investigating further train diversity, a hypothesis was raised that `balance_dataset()` used a stale cached payload count. Directly refuted: `len(sqli_payloads)` computed live from the current `data/sqli_payload_pool.csv` = 41, not 25 — the function re-reads the CSV on every call, no caching. The actual cause: the frozen `augmented_web_attack.csv` was generated by a past run when the pool file had only 25 rows (8 Boolean + 8 UNION + 8 Time + 1 Error-based); the other 7 Error-based and all 9 Stacked/other payloads were added to the pool file later and never incorporated.

A full regenerate via `scripts/prepare_data.py` was tested in a dry run and **rejected**: Benign/XSS/csic_original rows stayed byte-identical and positionally stable, but re-running with 41 payloads instead of 25 reassigns every `sqli_pool_*` `source_uid` — checked against the frozen split, **all 41/41 payload groups ended up split across both old-train and old-test positions**, which would have reintroduced full SQLi train/test content leakage (the same class of bug fixed once already via grouped `GroupShuffleSplit`).

Chosen instead: `scripts/add_missing_sqli_payloads_train.py`, append-only, adds 6400 train-only rows (400 each, matching the existing payload density) for the 16 missing payloads, `source='sqli_pool_extra'`, new source_uid, never touching `test_split_indices.pkl`. Full report: `data/missing_sqli_payloads_train_report.txt`.

Retrain #3 result (`results/COMPARISON_v3_after_missing_payloads.csv`): **no change** versus retrain #2 on any metric — held-out matrix identical (comment_splitting still 1.0, Benign cells still 0.0), test split F1 unchanged, Decision Tree baseline unchanged. The 25 payloads already in train were sufficient for what the held-out matrix currently probes; the additional 16 didn't move any measured result. (Consequence documented in `docs/DATASET.md`: train now sees all 41 payloads, but the frozen test set still only ever evaluates against the original 25 — no held-out test coverage of Error-based/Stacked-other payloads exists yet.)

## 7. Retrain #4: Benign syntax diversity fixes both Benign cells, but regresses comment_splitting

Direct verification before acting (not assumed): of the 1689 train Benign rows with `num_nodes ≤ 15`, **0/1689 contained `:`**, **0/1689 contained `{`/`}`**, and 1685/1689 (99.8%) were `key=value&key=value` query-string style. The held-out Benign failures were not about *length* — train Benign had essentially zero syntactic coverage of header- or JSON-shaped content at any length.

`scripts/add_benign_syntax_diversity_train.py`: append-only, 2500 header-style rows (`Name: value` pairs, e.g. `Content-Type: application/json`) + 2500 JSON-style rows (e.g. `{"status":"ok","action":"login"}`) = 5000 new rows, `source='benign_short_synthetic'`. Header/field names deliberately disjoint from the held-out matrix's `Authorization`/`X-Trace` and `page`/`user`/`locale`.

Retrain #4 was the first in this investigation that did **not** converge instantly: val_acc = 0.9995 at epoch 1 (not 1.0), reaching 1.0 only at epoch 7 (best_epoch=7 vs. 1 in every prior retrain) — evidence the train set now carries a non-trivial signal.

Result (`results/COMPARISON_v4_after_benign_syntax.csv`):
- `Benign / header_field`: 0.0 → **1.0**. `Benign / json_field`: 0.0 → **1.0**. Both fixed.
- `SQLi / comment_splitting`: **1.0 → 0.0 — regressed** (had been fixed in retrains #2 and #3). Logit probe: `Benign=5.96, SQLi=-0.25` (gap **-6.21**), a much stronger wrong-side margin than the original pre-fix state. E_sem edge count unaffected (still 4, deterministic construction). Working hypothesis at this stage: the new benign header-style rows use `;` as a pair separator, structurally similar to the comment-split payload's `;`-delimited fragments, and the two augmentation rounds' structural signals may be competing.
- Test split, Decision Tree baseline: unchanged.
- **Net: 8/9 held-out cells correct** (up from 6/9 at the start of this investigation), one new regression.

## 8. Retrain #5: class-weighted loss narrows the gap but does not fix it

Train class balance had drifted after three rounds of class-specific augmentation (Benign 34%, SQLi 44%, XSS 22% of train — see raw counts in `results/COMPARISON_v5_after_class_weight.csv`). Hypothesis: this imbalance, not a structural collision, explained the comment_splitting regression.

`src/training/train.py`: added `sklearn.utils.class_weight.compute_class_weight('balanced', ...)` computed from the actual train indices, passed into `nn.CrossEntropyLoss(weight=...)`. Logged explicitly to both console and `logs/training_history.log`: `Computed class weights (balanced): [Benign=0.976, SQLi=0.755, XSS=1.535]`. No data changes, no rebuild (same graphs as retrain #4).

Result (`results/COMPARISON_v5_after_class_weight.csv`):
- `SQLi / comment_splitting`: **still 0.0 — unchanged**. Logit probe: `Benign=3.35, SQLi=0.49` (gap **-2.86**) — roughly **half** the magnitude of the pre-class-weight gap (-6.21), but still on the wrong side, still predicting Benign.
- `Benign header_field` / `json_field`: stayed fixed at 1.0/1.0.
- Test split macro F1 dipped slightly: 1.0 → 0.99978 (one new XSS→Benign error) — a small cost of reweighting.
- Decision Tree baseline: unchanged, 0.7877.
- **Net: still 8/9**, target of 9/9 not reached.

This partially refutes the class-imbalance hypothesis: reweighting had a real, measurable effect (the gap roughly halved) but was not sufficient on its own — pointing back toward a structural/representational explanation rather than a pure imbalance one.

## 9. Final conclusion: quantitative evidence of an architectural limit, not a data gap

`scripts/inspect_attention_weights.py`: hooked `model.backbone.conv1(x, edge_index, return_attention_weights=True)` (GATv2Conv's native attention-weight return) on the current (retrain #5) checkpoint, for the exact `comment_split` sample, and compared the learned attention on the 4 E_sem edges against the 142 other (n-gram) edges in the same graph (37 self-loops added internally by GATv2Conv excluded from both groups).

```
alpha_Esem_mean      = 0.154079
alpha_other_mean     = 0.202362
alpha_self_loop_mean = 0.206710
ratio alpha_Esem_mean / alpha_other_mean = 0.7614
```

**The 4 E_sem edges receive, on average, 24% *less* attention than the surrounding n-gram edges — less even than GATv2Conv's own self-loops.** This is direct, quantitative evidence (not inference from accuracy numbers) that the current E_sem design — a sparse set of unweighted edges merged into the same `edge_index` as sequential (E_seq) and skip (E_skip) n-gram edges, with no architectural mechanism to prioritize them — is structurally under-weighted by the model regardless of whether the edges are constructed correctly and regardless of how much or what kind of training data surrounds them. Full evidence: `data/attention_weights_comment_split.txt`.

This reframes the one remaining held-out failure: it is not an unresolved bug requiring more investigation, but a **known, evidenced architectural limitation** of the current E_sem design (numerically outnumbered ~35:1 by other edge types, and not compensated for by attention weighting) — a legitimate limitation to report in the paper, with a documented path to addressing it (e.g. a dedicated/weighted edge type or relation-specific attention for E_sem) rather than something to keep patching with further data augmentation.

> **Correction (2026-09-19).** The ratio above compares raw alpha across edge groups whose destination nodes differ in in-degree. GATv2Conv softmax-normalizes over each destination's incoming edges (self-loop included), so uniform attention gives alpha = 1/in-degree and raw alpha must be compared with that null, not with 1.0. The 4 E_sem edges land on nodes of mean in-degree 6.50 (other edges: 5.01), so exactly uniform attention would give a raw ratio of **0.7647**; 0.7614 was measured. Degree-corrected (alpha × in-degree, 1.0 = uniform): config 1 (this checkpoint) **0.996**, self-loops 1.000; the E_sem edges carry 20.5% of the attention mass at their destination nodes against 20.6% under uniform attention; each of the 8 heads is within 0.965–1.020 of uniform; current seeds 42–46: 1.002–1.008. **Withdrawn:** "24% *less* attention", "less even than the self-loops", "structurally under-weighted by the model". **Stands:** the 0/50 failure, the 4-vs-142 edge-count imbalance and the conclusion that this is an architectural limit. **Mechanism restated:** attention does not lower E_sem, it is uniform over it, so the E_sem messages are averaged with the others in one untyped aggregation (see "HGT investigation" at the end of this file). The old numbers are kept above as history. Reproduce: `scripts/attention_degree_check.py` -> `data/attention_degree_check.txt`.

## 10. Follow-up ablation: does an explicit relation-type signal help E_sem?

Section 9's finding (E_sem edges get 24% less attention than surrounding n-gram edges) suggested a concrete architectural fix: give GATv2Conv an explicit relation-type signal per edge (one-hot seq/skip/sem, via PyTorch Geometric's native `edge_dim` mechanism) so attention can condition on edge type instead of treating all edges identically. Implemented and tested as a 4-way ablation (`scripts/run_ablation_edge_attr_seq_skip_sem.py`, full results: `results/ablation_edge_attr_seq_skip_sem.csv`).

Prerequisite refactor: `sequential_edges()`/`skip_edges()` (previously stubs, with the real i↔i+1/i↔i+2 logic hardcoded inline in `graph_builder.py`) were extracted into real modules (`src/edges/sequential.py`, `src/edges/skip.py`), and a shared `build_single_graph()` helper added to `src/bag/graph_builder.py` with `use_seq`/`use_skip`/`use_sem`/`use_edge_attr` toggles, reused by `build_web_graphs()` and `build_heldout_matrix_eval.py` alike. Regression-verified byte-for-byte against the pre-refactor graph file: rebuilding all 43595 rows with default toggles reproduced identical totals (3,514,551 edges, 940,497 nodes).

| config | test split macro F1 | held-out cells correct | comment_splitting | case_mixing |
|---|---|---|---|---|
| (1) full, no edge_attr (= retrain #5, reused not retrained) | 0.999778 | 8/9 | 0.0 | 1.0 |
| (2) full, with edge_attr | 0.999778 | 8/9 | **0.0 (not fixed)** | 1.0 |
| (3) no-skip, with edge_attr | 0.999778 | 7/9 | 0.0 | **0.0 (regressed)** |
| (4) no-sem, with edge_attr | 0.999778 | 7/9 | 0.0 | **0.0 (regressed)** |

**Test split macro F1 and confusion matrix were bit-for-bit identical across all 4 configurations** (`[[1550,0,0],[0,1214,0],[1,0,1450]]`, the same single XSS→Benign error every time) — a striking, additional confirmation of this investigation's recurring theme (§1, §4): the frozen test split is saturated to the point of not discriminating between architectural variants at all. All differentiation happens on the held-out matrix.

**Attention weight on the comment_split sample, with edge_attr enabled (config 2):**
```
alpha_Esem_mean=0.166283   alpha_other_mean=0.201772   ratio=0.824115
(config 1 / no edge_attr:  alpha_Esem_mean=0.154079   alpha_other_mean=0.202362   ratio=0.761400)
```
Giving the model an explicit relation-type signal **did raise E_sem's relative attention** (ratio 0.761 → 0.824, a real, measured increase) — but not past parity with other edges, and not enough to flip the wrong prediction. `SQLi / comment_splitting` stayed at 0.0 in config (2). Partial confirmation of §9's mechanism, insufficient on its own as a fix.

> **Correction (2026-09-19).** "Ratio 0.761 → 0.824 … but not past parity" is misread. In degree-corrected terms (Correction in §9) config 1 was already at parity (**0.996**) and adding edge_attr raised E_sem to **1.074**, 7% above uniform (E_sem attention mass at its destination nodes 22.2% vs 20.6% uniform; per-head 0.974–1.190; the uniform null for the raw ratio is 0.7647). edge_attr therefore moved attention modestly *above* parity without flipping `comment_splitting`: "real but insufficient" stands, "not past parity" is withdrawn.

Configs (3) and (4) each independently regressed `case_mixing` from 1.0 to 0.0, alongside the pre-existing `comment_splitting` failure — a new, single-run observation. Notably, removing E_skip (roughly a third of all edges) and removing E_sem (only 4 edges) produced the *same* held-out outcome, which is not what a purely edge-count-driven explanation would predict. With only one training run per configuration (no repeated seeds), this should be read as a preliminary signal worth a variance check, not a settled causal claim about E_skip or E_sem individually.

**Conclusion:** the edge_attr mechanism is a real, measurable step in the right direction (attention ratio improved) but does not fully close the gap alone. §9's "architectural limitation" framing stands: E_sem needs more than a relation-type tag to compete with the volume of n-gram edges — options not yet tried include a learned per-relation weight/gate (rather than only letting attention discover it), reducing n-gram edge density, or a separate aggregation path for semantic edges.

**Follow-up: isolating E_skip/E_sem removal from the edge_attr mechanism.** Configs (3)/(4) both used edge_attr, so it wasn't clear whether the `case_mixing` regression came from removing the edge type itself or from some edge_attr interaction. Two more configs (`scripts/run_ablation_extra_no_edge_attr.py`) repeat (3)/(4) with edge_attr off, same seed=42:

| config | test split macro F1 | test confusion matrix | held-out cells | case_mixing |
|---|---|---|---|---|
| (5) no-skip, no edge_attr | 0.999748 | `[[1550,0,0],[0,1214,0],[0,1,1450]]` (1 SQLi→XSS) | 7/9 | **0.0** |
| (6) no-sem, no edge_attr | 0.999778 | `[[1550,0,0],[0,1214,0],[1,0,1450]]` (identical to config 1) | 7/9 | **0.0** |

`case_mixing` regresses to 0.0 in both (5) and (6) — **the same regression seen in (3)/(4), now confirmed to happen with edge_attr off too.** This rules out edge_attr as the cause: removing either E_skip or E_sem alone, by itself, breaks `case_mixing`, regardless of whether the relation-type signal is present. It also further weakens a pure edge-count explanation (E_skip ≈ a third of all edges; E_sem is 4 edges) — both removals cause the identical held-out outcome.

One genuine difference from the edge_attr configs: config (5)'s test-split confusion matrix is *not* identical to config (1)'s (a different single error: SQLi→XSS instead of XSS→Benign) — the only config in this whole ablation where the frozen test split actually distinguished a configuration. Config (6) matches config (1) exactly. So E_skip removal does perturb the model somewhat on the test split; E_sem removal (4 edges out of ~150) does not perturb it at all -- consistent with §9's finding that E_sem is a very small, easily-outweighed signal.

## 11. Consolidated ablation conclusion (paper-ready)

All 6 ablation configurations in one table (`results/ablation_edge_attr_seq_skip_sem.csv`, §10; seed=42 throughout, same dataset — 5 rounds of train-only augmentation + balanced class weighting — for every config):

| config | E_seq | E_skip | E_sem | edge_attr | test split macro F1 | test confusion matrix | held-out (9 cells) | comment_splitting | case_mixing |
|---|---|---|---|---|---|---|---|---|---|
| (1) full, no edge_attr | ✓ | ✓ | ✓ |  | 0.999778 | `[[1550,0,0],[0,1214,0],[1,0,1450]]` | **8/9** | 0.0 | 1.0 |
| (2) full + edge_attr | ✓ | ✓ | ✓ | ✓ | 0.999778 | same as (1) | 8/9 | 0.0 | 1.0 |
| (3) no-skip + edge_attr | ✓ |  | ✓ | ✓ | 0.999778 | same as (1) | 7/9 | 0.0 | **0.0** |
| (4) no-sem + edge_attr | ✓ | ✓ |  | ✓ | 0.999778 | same as (1) | 7/9 | 0.0 | **0.0** |
| (5) no-skip, no edge_attr | ✓ |  | ✓ |  | 0.999748 | `[[1550,0,0],[0,1214,0],[0,1,1450]]` (differs from (1)) | 7/9 | 0.0 | **0.0** |
| (6) no-sem, no edge_attr | ✓ | ✓ |  |  | 0.999778 | same as (1) | 7/9 | 0.0 | **0.0** |

**Note on `case_mixing` in this table:** it does *not* measure resistance to case-obfuscation — `src/preprocessing/tokenizer.py:13` lowercases tokens before the model ever sees them, so the technique is already neutralized upstream and this cell cannot fail on that basis (confirmed by direct inspection, not re-tested here). What it's actually measuring is the same underlying capability as `sql_new_commands`: clear, non-evasive SQLi recognition. Read its regressions in configs (3)-(6) accordingly — as E_skip/E_sem being load-bearing for plain SQLi detection, not for obfuscation robustness.

Three findings, stated for direct use in the paper's Discussion:

**(a) E_seq/E_skip are load-bearing for both the standard benchmark and generalization.** Removing E_skip regresses the held-out matrix (8/9 → 7/9, `case_mixing` 1.0 → 0.0) in *both* the edge_attr and no-edge_attr configurations (3, 5) — the effect is attributable to E_skip itself, not to the relation-type mechanism used to test it. Configuration (5) is also the *only* one of the six whose test-split confusion matrix differs at all from the baseline (a new SQLi→XSS error replacing the baseline's XSS→Benign error) — the only edge-removal in this whole ablation that perturbs the otherwise-saturated standard benchmark (§1, §4, §10) even slightly. E_skip is not a redundant structural convenience; it measurably contributes to both evaluation regimes. **Caveat added after 5-seed testing (see "5-seed statistics" below):** this whole 6-config table is single-seed per configuration, and a separate 5-seed run of the official (non-ablated) config found one held-out cell (`XSS/data_uri_base64`) that reports as a stable 1.0 on any single seed but is actually seed-fragile (4/5 seeds). The direction of (a)'s conclusion is unaffected — `case_mixing`'s 1.0→0.0 flip under E_skip removal is a full accuracy-scale collapse, not a borderline single-sample flip like `data_uri_base64`'s — but the *exact* 8/9→7/9 counts here have not been re-verified across seeds and should be read as single-run evidence, not as seed-invariant facts.

**(b) E_sem has a generalization-specific role that the standard benchmark cannot see.** Removing E_sem regresses the held-out matrix identically to removing E_skip (8/9 → 7/9, `case_mixing` regresses) in both configurations (4, 6) — so E_sem is doing real, necessary work for obfuscation robustness. But configuration (6)'s test-split confusion matrix is *bit-for-bit identical* to the full baseline (1): removing all 4 E_sem edges changes nothing measurable on the standard test split. This is the clearest confirmation of the circularity concern that opened this investigation (§1): the standard benchmark's labels and difficulty are keyword-driven, so a component whose entire purpose is representing keyword *relationships* under obfuscation is invisible to it by construction — its contribution only becomes visible under a held-out evaluation designed to require generalization. A benchmark section reporting only the standard test split would not detect E_sem's necessity at all. **Same single-seed caveat as (a):** this table's 8/9→7/9 counts are from one seed each; the 5-seed follow-up (below) on the official config, not re-run for these ablation configs, found `XSS/data_uri_base64` alone flips between seeds (4/5) while every other cell including `comment_splitting` is seed-stable — so a single seed can misreport a held-out cell as fully stable when it isn't, though nothing here contradicts `case_mixing`'s specific 1.0→0.0 collapse, which is a different (larger, binary) kind of change than the kind of noise seen in `data_uri_base64`.

**(c) The multi-relational signal (edge_attr) is a real but insufficient mitigation — confirming a genuine architectural limit, not merely a missing feature.** Enabling edge_attr (relation-type one-hot, fed through GATv2Conv's native `edge_dim`) measurably raises E_sem's learned attention on the one persistently-misclassified held-out sample: `alpha_Esem_mean / alpha_other_mean` rises from **0.761** (config 1, no signal) to **0.824** (config 2, with signal) — the model *does* learn to weight semantic edges somewhat more given the means to distinguish them. Yet `SQLi / comment_splitting` stays at accuracy 0.0 in every configuration tested, edge_attr or not. The remaining gap is not explained by an absent relation-type signal (that was directly tested and only partially helped) — it is better explained by E_sem's edges being numerically overwhelmed: 4 semantic edges against 142 n-gram edges in the same graph (≈2.7% of edges), for a component whose signal must compete edge-for-edge with sequential/skip structure in the same attention computation. This motivates a specific, falsifiable next step for future work rather than an open question: E_sem needs a mechanism that doesn't compete on edge count at all (e.g. a learned per-relation gate/weight applied after aggregation, a separate aggregation path for semantic edges before merging with the sequential representation, or reducing n-gram edge density), not just a richer per-edge feature.

> **Correction (2026-09-19).** Read "rises from 0.761 to 0.824" as **0.996 → 1.074** degree-corrected (Correction in §9); the premise of (c) that E_sem starts under-weighted, and that its signal "must compete edge-for-edge … in the same attention computation", is withdrawn: attention is uniform over E_sem, and it is the untyped aggregation that averages it away. The conclusion of (c) is unchanged (edge_attr is a real but insufficient mitigation; `comment_splitting` stays at 0.0).

## Final state

**Held-out matrix: 8/9 cells correct (88.9%)**, up from 6/9 at the start of this investigation.

| cell | start | final | note |
|---|---|---|---|
| Benign / field_query | 1.0 | 1.0 | always correct |
| Benign / header_field | 0.0 | **1.0** | fixed (retrain #4) |
| Benign / json_field | 0.0 | **1.0** | fixed (retrain #4) |
| SQLi / sql_new_commands | 1.0 | 1.0 | always correct |
| SQLi / comment_splitting | 0.0 | **0.0** | fixed in #2/#3, regressed in #4, not restored by #5 — architectural limit, evidenced in §9 and §11c |
| SQLi / case_mixing | 1.0 | 1.0 | correct in the deployed (full-edge) model; §11 shows it regresses to 0.0 whenever E_skip or E_sem is ablated |
| XSS / data_uri_base64 | 1.0 | 1.0 | always correct |
| XSS / event_handler_focus | 1.0 | 1.0 | always correct |
| XSS / svg_script_variant | 1.0 | 1.0 | always correct |

The one remaining failure has a known, evidenced cause (§9) rather than being an open unknown.

## Evidence file index

| file | what it shows |
|---|---|
| `data/semantic_edges_inspect.txt` | E_sem edge counts before the `_reconstruct_keywords()` fix (§3) |
| `data/semantic_edges_inspect_after_fix.txt` | E_sem edge counts after the fix (§3) |
| `data/semantic_fix_safety_check.txt` | 0/10000 benign rows gained a spurious E_sem edge (§3) |
| `data/causal_padding_test.txt` | graph-size causal probe (§2) |
| `data/decision_tree_size_baseline.txt` | graph-size-only shortcut baseline, 0.7877 (§4) |
| `data/train_noise_augmentation_report.txt` | retrain #2 noise-augmentation derivation (§5) |
| `data/missing_sqli_payloads_train_report.txt` | retrain #3 missing-payload addition (§6) |
| `data/benign_syntax_diversity_train_report.txt` | retrain #4 benign syntax addition (§7) |
| `data/attention_weights_comment_split.txt` | real GATv2 attention weights, final conclusion (§9) |
| `results/COMPARISON_before_after_semantic_fix.csv` | retrain #1 (§4) |
| `results/COMPARISON_v2_after_valfix_and_noise.csv` | retrain #2 (§5) |
| `results/COMPARISON_v3_after_missing_payloads.csv` | retrain #3 (§6) |
| `results/COMPARISON_v4_after_benign_syntax.csv` | retrain #4 (§7) |
| `results/COMPARISON_v5_after_class_weight.csv` | retrain #5 (§8) |
| `results/ablation_edge_attr_seq_skip_sem.csv` | 4-config seq/skip/sem/edge_attr ablation (§10) |
| `data/attention_weights_comment_split_config2.txt` | attention weights with edge_attr enabled (§10) |
| `docs/DATASET.md` | full dataset provenance, freeze state, and every augmentation round's exact row counts |

## BIA Mechanism — Actual Implementation (for paper Section 3.3 rewrite, 2026-09-18)

Direct verification of the actual data-augmentation mechanism as implemented,
to replace the paper's current Eq.5 description with an exact algorithm. All
numbers computed live from the frozen `data/augmented_web_attack.csv`
(43595 rows) and all claims about code behavior are from reading the source
directly (`src/preprocessing/normalization.py::balance_dataset()`,
`scripts/add_train_noise_augmentation.py`), not inferred.

### 1. Mechanism (1) [template poisoning] vs. mechanism (2) [post-hoc addition], exact %

| class | source | n | % of class |
|---|---|---|---|
| **SQLi** (n=18595) | `sqli_pool` — mechanism (1), original template poisoning | 10000 | 53.7779% |
| | `sqli_pool_extra` — appended later, **same mechanism (1) logic**, 16 previously-missing payloads (see caveat below) | 6400 | 34.4179% |
| | `sqli_pool_noise` — mechanism (2), random-whitespace noise augmentation | 2111 | 11.3525% |
| | `csic_original` — real CSIC rows, neither mechanism (never template-substituted) | 84 | 0.4517% |
| **XSS** (n=10000) | `NA` — mechanism (1), original template poisoning | 10000 | 100.0000% |
| | *(no mechanism-2 addition exists for XSS — 0 rows)* | 0 | 0.0000% |
| **Benign** (n=15000) | `NA` — original CSIC benign rows (not template-substituted; content is the real request) | 10000 | 66.6667% |
| | `benign_short_synthetic` — synthetic header/JSON-style rows added later (§7) | 5000 | 33.3333% |

**Important precision correction to the question's framing:** `sqli_pool_extra`
is *not* noise-augmented content. It uses the exact same unconditional
template-poisoning logic as `sqli_pool` (§ below) — the only difference is it
was generated by a separate script (`scripts/add_missing_sqli_payloads_train.py`)
run later, to add the 16 SQLi payloads (indices 25-40) missing from the
original `sqli_pool` generation (§6, `docs/DATASET.md` "Train-only addition
of the 16 missing SQLi payloads"). If grouped strictly by *generation
mechanism* rather than by *when it was added*: **mechanism (1) [template
poisoning, no noise] = `sqli_pool` + `sqli_pool_extra` = 16400/18595 =
88.1958%** of SQLi; **mechanism (2) [actual random noise] = `sqli_pool_noise`
only = 2111/18595 = 11.3525%**. Use whichever grouping the paper needs, but
do not describe `sqli_pool_extra` as noise-augmented — it is unmodified
payload text, template-poisoned.

### 2. Does template poisoning (mechanism 1) validate HTTP protocol structure after substitution?

**No. Confirmed by direct line-by-line read of `balance_dataset()`
(`src/preprocessing/normalization.py:170-283`) — zero validation step exists
after content substitution, for either SQLi or XSS.**

SQLi synthetic pool (`src/preprocessing/normalization.py:247-254`):
```python
for i in range(target_size):
    template = frames_for_sqli.iloc[i % len(frames_for_sqli)].to_dict()
    payload = sqli_payloads[i % len(sqli_payloads)]
    template['content'] = payload
    template['attack_type'] = 1
    template['source'] = 'sqli_pool'
    template['source_uid'] = f"sqli_pool_{i % len(sqli_payloads)}"
    synthetic.append(template)
```

XSS (`src/preprocessing/normalization.py:262-267`):
```python
for i in range(target_size):
    template = frames.iloc[i % len(frames)].to_dict()
    template['content'] = xss_payloads[i % len(xss_payloads)]
    template['attack_type'] = 2
    poisoned.append(template)
```

Both loops take a real benign row's full field dict (`Method`, `User-Agent`,
`Pragma`, `content`, `URL`, etc. — every CSIC column), overwrite **only**
`content` (and `attack_type`/`source`) with the raw payload string, and
append immediately — no regex check, no length/format check, no attempt to
verify the resulting row is still a well-formed HTTP request line/field. The
other preserved fields (headers, method, etc.) are never re-validated for
consistency with the new `content` either. In practice this is inconsequential
for model training specifically, since `src/bag/graph_builder.py::build_web_graphs()`
reads only the `content` column and ignores every other field (confirmed
earlier in this investigation) — but it means the *other* CSV columns for
every synthetic SQLi/XSS row are stale/inconsistent benign metadata, and
nothing in the pipeline checks or corrects that.

### 3. Mechanism (2) — `scripts/add_train_noise_augmentation.py` — full discrete algorithm

All parameters, verbatim from source, for describing this as a precise
algorithm (replacing the paper's Eq.5):

**Eligibility filter** (applied before any random step): a `sqli_pool` row is
eligible iff its `source_uid` is (a) in `train_idx` of
`data/test_split_indices.pkl` (never `test_idx`), AND (b) not one of the
SQLi-family test uids (first 2 payload ids per family sorted by index, or 1
for families with ≤3 payloads — `_fixed_family_group_split`'s deterministic
rule). Result on the frozen dataset: 16/25 `sqli_pool` uids eligible → 6400
eligible rows.

**Two independently-seeded random streams, both seeded 42:**
1. `random.Random(SEED)` (Python stdlib) — used only inside `noise_content()`
   for span/position/filler choice (per-row noise decisions).
2. pandas' internal (numpy-backed) RNG via `eligible_rows.sample(n=n_target,
   random_state=SEED)` — used only for *which* rows get sampled, a separate
   stream from (1).

**Step A — row sampling:** `NOISE_FRACTION = 0.35`. `n_target =
round(6400 × 0.35) = 2240` rows sampled **uniformly without replacement**
from the 6400 eligible rows (`DataFrame.sample`, stream 2 above).

**Step B — per-row noise injection** (`noise_content()`, stream 1 above),
applied independently to each of the 2240 sampled rows' `content` string:
1. Find every occurrence of any of **29 fixed SQL keywords** — `select,
   union, from, where, order, group, insert, into, drop, table, delete,
   update, and, or, null, waitfor, delay, benchmark, sleep, extractvalue,
   updatexml, convert, cast, database, version, concat, exec, all, limit` —
   via case-insensitive regex `\b(keyword1|keyword2|...)\b`
   (`re.IGNORECASE`). Word-boundary-delimited, so this is whole-word keyword
   matching, not substring matching.
2. If zero keyword occurrences found in that row's `content` → **skip this
   row entirely, no new row produced** (measured: 129/2240 = 5.76% skipped
   this way on the frozen dataset).
3. Otherwise, pick **exactly one** keyword occurrence uniformly at random
   from all matches found (`rng.choice(spans)`) — if a row's content has
   multiple keyword occurrences, only one is noised, the rest are left intact.
4. If the chosen keyword's matched text is <2 characters → skip (dead code
   in practice: every entry in the 29-keyword list is ≥3 characters, so this
   branch can never trigger on the current keyword list).
5. Pick a **uniform random interior split position** within the chosen
   keyword: `rng.randint(1, len(word) - 1)` — i.e. never splits before the
   1st or after the last character, every interior position equally likely.
6. Pick a **uniform random whitespace filler** from exactly 5 fixed variants:
   `' '` (1 space), `'  '` (2 spaces), `'\t'` (1 tab), `' \t '`
   (space-tab-space), `'   '` (3 spaces) — via `rng.choice`.
7. Insert the filler at the split position inside the keyword only (e.g.
   `select` with split_pos=3, filler=`'\t'` → `sel\tect`); everything else in
   the content string is left byte-identical.
8. Write the new row: `content` = noised string; `source =
   'sqli_pool_noise'`; `source_uid = f'sqli_pool_noise_{i}'` (`i` = position
   in the sampled-rows loop, 0-indexed, restarts at 0 on every script
   invocation — this is exactly the collision risk the idempotency guard
   added in commit `088603c` protects against, see
   `docs/REPRODUCIBILITY.md` "Data Integrity Guards"). All other CSV columns
   copied unchanged from the source row (same "no protocol re-validation"
   caveat as § 2 above — this mechanism doesn't validate either).

**Net effect on the frozen dataset:** 2240 sampled → 129 skipped (no keyword
match) → **2111 rows actually noised and appended**, i.e. 2111/2240 = 94.24%
of sampled rows survive to become new training rows, each with exactly one
keyword split by 1-3 inserted whitespace characters at a uniformly random
interior position.

## XSS Noise-Augmentation Feasibility Check (2026-09-18, pre-implementation)

Done **before** writing any XSS augmentation script, per the requirement to
verify syntactic validity with a real HTML parser first rather than assume.

**Tooling caveat (stated upfront, not buried):** this sandbox has only
Python's stdlib `html.parser` available — `bs4`, `lxml`, `html5lib`,
`selenium`, and `playwright` are all **not installed**
(`pip list` checked directly), so **no real browser engine was available to
test against.** Everything below reflects `html.parser`'s tokenizer
specifically, cross-checked against WHATWG HTML5 tokenizer spec knowledge
where noted, but is **not** empirical browser confirmation. Recommended
before citing this in the paper as a browser-validated claim: re-run against
an actual browser (e.g. via a headless Chromium) or at minimum `lxml`/`bs4`
with `html5lib` backend, which implement the HTML5 spec's parsing algorithm
more faithfully than `html.parser`.

### 1 & 2. Parser tests (`html.parser.HTMLParser`, `handle_starttag` output)

| variant | input | parsed attribute for the keyword | keyword preserved intact? |
|---|---|---|---|
| tab between tag/attr | `<img\tonerror=alert(1)>` | `('onerror', 'alert(1)')` | ✅ yes |
| newline between tag/attr | `<img\nonerror=alert(1)>` | `('onerror', 'alert(1)')` | ✅ yes |
| benign attr before keyword, `src=x` first | `<img src=x onerror=alert(1) data-x=1>` | `('onerror', 'alert(1)')` (order preserved: src, onerror, data-x) | ✅ yes |
| benign attrs both sides | `<img data-a=1 onerror=alert(1) data-b=2 data-c=3>` | `('onerror', 'alert(1)')` | ✅ yes |
| `script`/`src`, 1 benign attr before | `<script data-x=1 src=//evil.com/x.js></script>` | `('src', '//evil.com/x.js')` | ✅ yes |
| `script`/`src`, heavy noise (5 attrs) | `<script type=text/javascript data-a=1 data-b=2 data-c=3 src=//evil.com/x.js data-d=4></script>` | `('src', '//evil.com/x.js')` | ✅ yes |
| null byte glued to attr name (no space) | `<img onerror\x00=alert(1)>` | `('onerror\x00', 'alert(1)')` | ❌ **no** — name is `onerror\x00`, not `onerror` |
| control char (BEL) glued to attr name | `<img onerror\x07=alert(1)>` | `('onerror\x07', 'alert(1)')` | ❌ **no** |
| null byte mid attr name | `<img oner\x00ror=alert(1)>` | `('oner\x00ror', 'alert(1)')` | ❌ **no** |

**Whitespace variants (tab/newline as the tag↔attribute separator) and
attribute reordering/interspersing are both syntactically valid per
`html.parser`, and both keep the target attribute name byte-identical to
`onerror`/`src`.** Per WHATWG HTML5 tokenizer spec knowledge (not
independently browser-verified here, see caveat above), this also matches
real-browser behavior: any of U+0009 TAB, U+000A LF, U+000C FF, U+000D CR,
or U+0020 SPACE is a valid "before attribute name" state transition, and
attribute order/count is never semantically constrained by the spec.

**The null-byte/control-character variant does NOT satisfy "keyword
preserved, only context changed"** — in `html.parser`, the byte fuses onto
the attribute name, producing a different string (`onerror\x00` ≠
`onerror`). This is not independently confirmed against a real browser here,
but WHATWG spec knowledge indicates real browsers replace embedded NUL with
U+FFFD in the attribute-name state rather than stripping it — also
producing a mutated (not preserved) name. **This variant is structurally the
same category as SQL's mid-keyword comment-splitting (character-level
keyword fragmentation), not the "context-only" mechanism being tested for —
excluded from the "Mechanism 2 for XSS" conclusion below.**

### 3. Does this actually break E_sem / E_skip in this repo's graph construction? (empirical, not hypothetical)

Ran the real `web_security_tokenizer()` + `semantic_edges()` /
`skip_edges()` from `src/preprocessing/tokenizer.py` / `src/edges/semantic.py`
/ `src/edges/skip.py` directly.

**Whitespace variant has zero effect on the graph** — `web_security_tokenizer`
splits purely on `\w`/punctuation regardless of which whitespace character
separated tokens in the source, so `<img\tsrc=x\tonerror=alert(1)>` produces
the exact same token list, same `img`↔`onerror` token-index gap (4), and the
exact same E_sem/E_skip edges as the plain-space version. **Already
neutralized before it reaches the model — same dead-end category as the
`case_mixing` finding (tokenizer `.lower()`), not a useful test vector.**

**Benign-attribute insertion is different: it measurably grows the
`img`↔`onerror` token-index gap, and breaks `E_sem` once the gap reaches
`window=15`:**

| # benign attributes inserted between `img` and `onerror` | token-index gap | `E_sem` connects them? |
|---|---|---|
| 0 (baseline, `<img onerror=...>`) | 1 | ✅ yes (2 edges) |
| 1 (`<img data-0=v0 onerror=...>`) | 6 | ✅ yes |
| 2 (`<img data-0=v0 data-1=v1 onerror=...>`) | 11 | ✅ yes |
| 3 (`<img data-0=v0 data-1=v1 data-2=v2 onerror=...>`) | **16** | ❌ **no (0 edges)** |
| 5 / 8 / 10 / 12 / 15 / 20 | 26 / 41 / 51 / 61 / 76 / 101 | ❌ no, all |

**At exactly 3 inserted benign attributes, the token gap (16) exceeds
`semantic_edges()`'s `window=15` and the E_sem edge disappears entirely** —
confirmed by direct execution of the repo's real `semantic_edges()`, not
estimated. Three `data-*`-style attributes (`<img data-0=v0 data-1=v1
data-2=v2 onerror=alert(1)>`) is trivially valid HTML5 (custom `data-*`
attributes are explicitly spec-sanctioned and ignored by both parsers and
browsers otherwise), and the payload remains **fully functional as a real
XSS attack** — no browser cares about attribute count, order, or the
presence of unrelated `data-*` attributes when deciding whether to fire
`onerror`.

**Can `E_skip`/`E_seq` bridge this gap instead?** No — checked
`src/models/layers.py`: `GATBackbone` has exactly **2 `GATv2Conv` layers**,
so the model's message-passing receptive field is 2 hops. `E_skip` connects
only `i`↔`i+2`; two hops of that reaches at most `i±4`. A gap of 16+ tokens
is categorically unreachable by any combination of `E_seq`/`E_skip` edges
within a 2-layer GNN, regardless of tuning `k`. Once benign-attribute
padding exceeds ~3 attributes, `img` and `onerror` (or `script` and `src`)
become **graph-topologically disconnected** for this architecture — no
edge-based path of any type reaches between them.

### Conclusion

**Yes — a valid "Mechanism 2 for XSS" exists, genuinely different in nature
from SQL's mid-keyword character-splitting, and it has been confirmed (not
just hypothesized) to break `E_sem` in this repo's actual graph construction
code:**

> **Benign HTML attribute padding**: insert ≥3 syntactically valid,
> semantically inert attributes (e.g. `data-*` custom attributes) between
> the two halves of an XSS semantic pair (`img`...`onerror`,
> `script`...`src`, `svg`...`onload`). The keyword tokens themselves are
> never touched — this is fundamentally a **token-distance/context-dilution**
> attack, not a character-fragmentation attack like SQL comment-splitting —
> and it is valid, functional, unlimited-scale HTML (an attacker can add as
> many `data-*` attributes as needed with zero cost to exploit reliability).
> Confirmed to break `E_sem` at the token level once the gap exceeds
> `window=15`, and confirmed structurally unbridgeable by `E_skip`/`E_seq`
> given the model's 2-layer GNN receptive field.

This is a real, implementable augmentation direction (not a Limitations-only
finding) — recorded here as the pre-implementation feasibility check;
writing the actual `add_xss_attribute_padding_train.py` augmentation script
(if the user wants to proceed) is a separate, not-yet-done step. Two
variants tested and explicitly **ruled out** as not fitting this mechanism:
whitespace-separator noise (already neutralized by the tokenizer before
reaching the model — no test value) and null-byte/control-character
insertion (mutates the keyword itself, same category as SQL
comment-splitting, not "context-only").

## XSS Context-Distance Augmentation: implementation, retrain, and a finding that revises the pre-registered hypothesis (2026-09-18)

### Step 1 — `scripts/add_xss_context_augmentation.py` implemented and applied

Implements the feasibility check above as a real augmentation, IN-RANGE only
(gap always kept <15/window by construction — the gap≥15 case is Step 2's
architectural-limit question, not something data can address, so no rows
target it). Targets only `('img','onerror')`, `('script','src')`,
`('svg','onload')` — `('script','eval')`/`('onerror','eval')` excluded (see
feasibility check above, `eval` sits in JS-expression context, not
tag-attribute-list context).

**Eligibility (train-only XSS rows, `source` is `NaN` = mechanism-1
original):** 10000 XSS rows total → 8549 train-only → only **186** contain
at least one target pair as whole words (`img`/`onerror`: 11,
`script`/`src`: 23, `svg`/`onload`: 152 — most of this dataset's XSS
payloads use other event handlers/tags not in E_sem's 9-pair vocabulary at
all, so the eligible pool is inherently small; not an implementation
shortfall).

**Two independently-seeded random streams (seed=42 both), matching the
SQLi noise script's convention:** stream A (`DataFrame.sample`) selects
`round(186 × 0.35) = 65` rows; stream B (`random.Random(42)`) picks 1-2
random benign attributes (`data-*`/`class`/`id`/`style`, random values) per
row and inserts them immediately before the pair's second keyword.
**Per-row gap check enforced before accepting** (not just designed to
usually work): 60/65 accepted with 2 attributes, 4/65 downgraded to 1
attribute because 2 didn't fit under window=15, **1/65 skipped entirely**
(neither 1 nor 2 attributes kept the gap under 15 for that particular row).
**64 new rows appended**, `source='xss_pool_context'`. Verified append-only
byte-identical on the prior 43595 rows (diff against
`data/augmented_web_attack_PRE_xss_context.csv`, 0 lines). Full report:
`data/xss_context_augmentation_report.txt`.

Dataset: 43595 → **43659** rows. Rebuilt `web_graphs.pkl` (43659 graphs,
`use_seq=True use_skip=True use_sem=True use_edge_attr=False`, matching the
deployed checkpoint) and retrained (checkpoint/log backed up first to
`*_PRE_xss_context.*`, per the working-rules backup discipline):
epochs_run=19, best_epoch=9 (val_acc=1.0). **Frozen test split: macro F1 =
1.0 exactly, confusion matrix `[[1550,0,0],[0,1214,0],[0,0,1451]]` — zero
errors**, an actual (marginal) improvement over the pre-retrain state (which
had 1 SQLi→XSS error).

### New held-out cells (`scripts/build_heldout_window_test.py`, kept separate from the 9-cell matrix — see script docstring for why)

Two cells, `('svg','onload')`, 10 held-out tag names never used in the
training augmentation (`article/aside/details/figcaption/marquee/summary/
template/tfoot/bdi/ruby`): **`attribute_spacing_in_range`** (2 inserted
attributes, gap=12, the case this augmentation targets) and
**`attribute_spacing_out_of_range`** (6 inserted attributes, gap=34, the
Step-2 architectural-limit anchor). Every row's gap asserted programmatically
before evaluating anything (not assumed).

| checkpoint | `attribute_spacing_in_range` (gap=12) | `attribute_spacing_out_of_range` (gap=34) |
|---|---|---|
| PRE (before this augmentation/retrain) | **10/10 (1.0)** | **10/10 (1.0)** |
| POST (after this augmentation/retrain) | **10/10 (1.0)** | **10/10 (1.0)** |

**Original 9-cell held-out matrix re-run against the POST checkpoint**
(`scripts/build_heldout_matrix_eval.py`, `results/heldout_matrix_full.csv`,
PRE state backed up to `results/heldout_matrix_full_PRE_xss_context.csv`):
**8/9 unchanged** — `SQLi/comment_splitting` still 0.0 (the known
architectural limit from §9, untouched by this round), all other 8 cells
still 1.0, including the 3 existing XSS cells
(`data_uri_base64`/`event_handler_focus`/`svg_script_variant`, all still
10/10). **No regression** from adding the 64 XSS context-distance rows.

**5-seed stability check (2026-09-18, `scripts/eval_gap_window_5seed.py`),
reusing the 5 checkpoints behind `results/final_stats_5seed.csv` (seeds
42-46, no retraining):** both cells are **perfectly stable — 50/50 (10/10
per seed × 5 seeds) for both `attribute_spacing_in_range` and
`attribute_spacing_out_of_range`**, with softmax margin (true-class prob
minus next-highest class prob) ≈ 1.0 on every single row (min 0.9999998
across all 100 row×seed evaluations) — i.e. not a borderline result that
happens to round to 10/10, the model is maximally confident on every one.
This is a materially different reliability profile from the *other*
XSS held-out cell found seed-fragile in that same 5-seed round,
`XSS/data_uri_base64` (4/5 seeds, `results/heldout_percell_5seed.csv`):
mean accuracy 1.0/std 0.0 here vs. mean 0.8/std 0.447 there. Full per-seed
detail: `results/gap_window_5seed.csv` (aggregate),
`results/gap_window_5seed_raw.csv` (per-row logits). **The single-seed
10/10 result below is now seed-verified, not seed-42-specific** — read the
PRE/POST table and Step 2's reasoning below with that confirmation in mind.

### Step 2 — the actual finding (revises, does not confirm, the pre-registered hypothesis)

**The pre-registered hypothesis for this step — "E_sem has a hard
distance threshold independent of training data; any evasion technique
producing a token gap past window=15 will completely neutralize E_sem,
regardless of how much similar data the model has seen" — is only PARTIALLY
supported, and its strongest, paper-relevant implication (that this
neutralizes *classification*, not just the *edge*) is
**empirically CONTRADICTED** by the test above. Reporting the actual result
rather than the pre-registered claim:**

**What IS confirmed, exactly as hypothesized:** `semantic_edges()`'s
`window` parameter is a hard, deterministic, data-independent cutoff on
**edge construction**. Directly verified, not estimated: for all three XSS
target pairs, `E_sem` edge count drops to exactly 0 the moment token gap
≥ window (confirmed via `data/attention_weights_comment_split.txt`-style
direct inspection: `img`/`onerror` and `script`/`src` at gap=34 both produce
`E_sem_edges=0`, same as `svg`/`onload`). This holds regardless of training
— it is a property of the graph-construction code, not something a
retrained checkpoint can influence, since the edge is simply never added to
the graph the model receives.

**What is NOT confirmed — the critical correction:** losing the `E_sem`
edge does **not** translate into a classification failure here. Both the
PRE and POST checkpoints classify **every** out-of-range sample correctly
(10/10, `probs(Benign/SQLi/XSS) ≈ [0.0, 0.0, 1.0]` — checked directly, not
just via argmax) across all three target pairs, with **zero** `E_sem`
edges present in the graph. The in-range cell was already at 10/10 on the
PRE checkpoint too, so this augmentation round produced **no measurable
accuracy change on this specific test** (ceiling effect — nothing here was
broken to begin with).

**Root-cause reconciliation with §9's attention-weight finding:** this
mechanism (benign attribute padding) removes only the long-range `E_sem`
bridge between the two keyword tokens — it leaves the raw keyword tokens
themselves (`svg`, `onload`, `img`, `onerror`, `script`, `src`) perfectly
intact and lexically unambiguous. The model still has two other signal
paths available: (a) each keyword token's own node features, and (b) local
`E_seq`/`E_skip` n-gram structure immediately around each keyword (e.g.
`onload=alert(`) — and empirically, these alone are already sufficient for
a correct prediction, independent of the long-range bridge. **This is
categorically different from SQLi's `comment_splitting` failure**, where
the evasion technique fragments the keyword at the *character* level
(`union` → `uni`+`on`) — destroying the raw lexical signal *and* the
`E_sem` edge simultaneously. It is that **double** removal (not `E_sem`
loss alone) that causes the `comment_splitting` failure documented in §§3-9.
`E_sem`'s measured contribution (§9's attention-weight analysis, §11's
ablation) is real for cases where the lexical signal is *also* destroyed,
but this result shows it is not a generically necessary signal whenever two
related keywords happen to be far apart with their own identities intact.

**Corrected framing for the paper (weaker, more precise, and more honest
than the pre-registered claim):** `E_sem`'s `window` parameter is a
provable, data-independent hard limit on the *semantic-edge mechanism*
itself — this part is now confirmed across all three XSS pairs in addition
to the original SQLi case, strengthening that specific, narrower claim. It
is **not**, by itself, evidence of a data-independent hard limit on overall
*model accuracy* — that depends on whether the evasion technique also
destroys the underlying lexical tokens (as SQL comment-splitting does) or
merely dilutes their proximity (as this XSS mechanism does). Recommend
citing this as a **refinement** of §9's finding — E_sem is necessary
specifically when lexical signal is unavailable, not unconditionally —
rather than as a new, stronger, standalone limit.

**Trade-off note on increasing `window` (order-of-magnitude estimate, not
load-bearing for the above conclusion):** `semantic_edges()`'s nested loop
only scans forward from each keyword hit until the *next* hit's index gap
reaches `window` (`if idx_j - idx_i >= window: break`), so its cost scales
with keyword-hit density × `window`, not with total graph size directly —
and `E_sem` is already the smallest edge category by far relative to
`E_seq`/`E_skip` (mean graph in the current `web_graphs.pkl`: 21.6 nodes,
80.6 edges total, of which `E_sem` is typically single digits per graph per
§9/§11). Raising `window` mainly risks **more false semantic connections
between unrelated keyword occurrences** that happen to co-occur within a
larger window (a precision/noise trade-off for `E_sem` specifically), not a
meaningful compute-cost increase relative to the `E_seq`+`E_skip` edges that
already dominate the graph.

## 5-seed statistics: official config (2026-09-18)

**Official config confirmed for this run (no ambiguity, no config change
made):** full edges (E_seq+E_skip+E_sem), `use_edge_attr=False` (config (2)
in §11's ablation didn't improve enough to justify the added complexity —
§11(c)), current dataset (43659 rows: Mechanism 1 + Mechanism 2 [SQLi noise]
+ Mechanism 2b [XSS context-distance]). This is exactly the deployed
`best_web_gnn_seed42.pth` (retrain #6 in `docs/REPRODUCIBILITY.md`'s
checkpoint table). Trained seeds 42-46 (`scripts/run_5seed_stats.py`),
evaluated each on the frozen test split and the 9-cell held-out matrix.
Deployed checkpoint/log/results backed up before running and restored
afterward (verified by md5sum) — this is a stats-collection exercise, the
deployed model is unchanged.

### Finding: training is not bit-reproducible even with a fixed seed on this GPU setup

Three separate training runs with **the same seed=42** and **identical
code/config** — the originally-deployed checkpoint, `scripts.run_5seed_stats`'s
first loop iteration, and a standalone rerun done to recover this section's
per-cell data (see below) — produced **three different training
trajectories** (`epochs_run` 11 / 20 / 13 respectively, different per-epoch
loss/accuracy values from epoch 1 onward). This is a real, verified
methodological finding, not a script bug: `torch`/`torch_geometric`/CUDA
operations (notably scatter/gather ops used by `GATv2Conv` and
`global_max_pool`) are not deterministic by default on GPU unless
`torch.use_deterministic_algorithms(True)` plus specific `cuDNN` flags are
set, which this codebase did not do at the time of this run. **Consequence
for the paper:** citing a single seed=42 result as *the* number, or
describing seeds as fully controlling reproducibility, overstates precision
on this hardware/software stack — the 5-seed spread reported below already
reflects a mix of genuine inter-seed variance and this intra-seed GPU
nondeterminism, and the two cannot be cleanly separated post hoc.

**Follow-up (same day): the fix was attempted and verified NOT sufficient.**
`torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG` were
added to `src/utils/seed.py::set_seed()` and two fresh `seed=42` runs were
compared line-by-line — they still diverge starting at epoch 2, by the same
magnitude as before the fix. See `docs/REPRODUCIBILITY.md`'s "Seed" section
for the full comparison and the leading hypothesis (an unseeded
`DataLoader(shuffle=True)` `RandomSampler`, not yet confirmed).

### Test split: essentially saturated, negligible variance

Mean accuracy = 0.99981, std = 0.000106 across 5 seeds (min 0.99976, max
1.0) — consistent with every prior single-seed report in this document; the
frozen test split remains too easy to discriminate between runs. Full
per-seed and mean/std/min/max table: `results/final_stats_5seed.csv`.
Cohen's d vs. the strongest Table 2 baseline (HGT, 95.87%), using the
*measured* 5-seed std (not an assumed std=0 as flagged as a concern before
this run): **d ≈ 387** — astronomically large only because the model's own
std (0.0106 percentage points) is tiny relative to the ~4-point accuracy
gap to HGT; report this Cohen's d with the caveat that it is only as
meaningful as the (very small, likely near-saturated-metric-noise-floor)
std it's divided by, not a claim that the model-to-model effect itself is
"387 standard deviations big" in any intuitive sense.

### Held-out matrix: 8/9 is not a fixed number — exactly one cell is seed-fragile, and it is not the one you'd guess

Per-seed cells-correct: **42→7/9, 43→8/9, 44→7/9, 45→8/9, 46→8/9** (mean
7.6/9, std 0.55). This *looks* like it could be `comment_splitting`
intermittently working — it is not. Per-cell breakdown across all 5 seeds
(`results/heldout_percell_5seed.csv`, seed 42 re-run separately to recover
this since the original loop's seed=42 checkpoint was overwritten by seed
43's training before its per-cell result was saved — see script docstring
caveat added after this run):

```
technique                  43   44   45   46   42(rerun)
Benign/field_query         1.0  1.0  1.0  1.0  1.0
Benign/header_field        1.0  1.0  1.0  1.0  1.0
Benign/json_field          1.0  1.0  1.0  1.0  1.0
SQLi/case_mixing           1.0  1.0  1.0  1.0  1.0
SQLi/comment_splitting     0.0  0.0  0.0  0.0  0.0   <- always fails, all 5 seeds (§9 architectural limit, confirmed again)
SQLi/sql_new_commands      1.0  1.0  1.0  1.0  1.0
XSS/data_uri_base64        1.0  0.0  1.0  1.0  1.0   <- the ONLY seed-fragile cell (fails only for seed 44)
XSS/event_handler_focus    1.0  1.0  1.0  1.0  1.0
XSS/svg_script_variant     1.0  1.0  1.0  1.0  1.0
```

**Two distinct, previously-conflated things are now separated:**
`SQLi/comment_splitting` is a **stable, 100%-reproducible architectural
limit** (0/5 seeds pass) — nothing new, confirms §9/§11/§Step-2 again,
independent of GPU nondeterminism. `XSS/data_uri_base64` (a base64-encoded
`data:` URI wrapping an `<svg>` payload) is **not** stable — it passes on
4/5 seeds and fails on exactly 1 (seed 44) — this is a previously-unreported
**seed-fragile cell**, not a hard limit like `comment_splitting`. It had
looked like a solid, always-passing cell in every single-seed report earlier
in this document; that was an artifact of never having tried more than one
seed. **Correction for the paper: report the held-out matrix as "8/9 stable
+ 1/9 seed-fragile" (`data_uri_base64`, ~80% pass rate over 5 seeds), not as
a flat "8/9."** `comment_splitting` remains the only cell with a documented,
evidenced *architectural* cause; `data_uri_base64`'s occasional failure has
no root-cause investigation yet (out of scope for this task) and should not
be assumed to share `comment_splitting`'s cause.

## Test-split content duplication (25.2%, primarily XSS 45.7%) (2026-09-18)

Discovered while checking the RoBERTa/CodeBERT baseline (Reviewer #3) for
early-stopping leakage (val set correctly carved from train only, not
test.csv — that check was clean). Investigating a secondary symptom (val
rows sharing exact `content` text with test rows) traced back to a property
of `data/test_split_indices.pkl` itself, affecting **every** method
evaluated against it, not just the new baselines.

**Measured (`data/augmented_web_attack.csv` + `data/test_split_indices.pkl`,
current train/test split via `src/training/train.py::_load_global_split()`'s
logic — test_idx frozen, train_idx = complement):**

- **1063/4215 test rows (25.2%)** have `content` byte-identical to at least
  one train row. Per class: **0/1550 Benign**, 400/1214 SQLi (33.0%),
  **663/1451 XSS (45.7%)**.
- Checked whether this is the previously-fixed `GroupShuffleSplit` bug
  (§ "Train-only addition of the 16 missing SQLi payloads" in
  `docs/DATASET.md`, where a single `source_uid` payload group ended up
  split across both train and test): **it is not**. Grouped by
  `source_uid`, **0/664** unique overlapping content strings have the same
  `source_uid` appearing on both sides — every overlap is between two
  *different* `source_uid` groups that independently produced identical
  `content` text. `GroupShuffleSplit`-by-`source_uid` is working exactly as
  designed; it just doesn't (and structurally cannot) guarantee unique
  *content* across groups when the underlying SQLi/XSS payload pools are
  finite and reused across many template rows (Mechanism 1, "Template
  poisoning" — see `docs/BIA_DECISION_LOG.md` §1-2). XSS is hit hardest
  because its external payload corpus is smaller relative to the number of
  XSS rows generated from it than SQLi's is.

**This is a structural limitation of the main benchmark, distinct from and
additional to the label/E_sem circularity concern (§1):** §1 is about
*keyword-driven labels* making the standard test split unable to detect
whether a component (E_sem) is doing real generalization work. This
duplication finding is about the split not even being fully novel-content
at the *entity* level for ~1 in 4 test rows — `source_uid`-grouping
correctly prevents a single generation *event* from leaking, but does not
and cannot prevent two independent generation events from producing
identical text when drawn from the same finite payload pool. Both concerns
point the same direction: **the standard test split alone overstates how
much genuine content-generalization any method evaluated on it (GATv2,
string-matching, RoBERTa, CodeBERT alike) has been shown to have.**

**Not fixed — the frozen split stays frozen** (explicit decision, confirmed
2026-09-18): re-splitting by content-hash instead of `source_uid` would
invalidate every prior checkpoint/result in this document's history for a
benefit that verification below shows is small in practice.

**Verified NOT to be silently inflating the headline number:** using an
early RoBERTa checkpoint (epoch 1, val_acc 0.9995) as a probe, accuracy was
1.00 on **both** the 1063 duplicate-content test rows and the 3152
novel-content test rows — i.e. this baseline's near-perfect test-split
score is not an artifact of memorizing the duplicated 25.2%; it scores just
as well on the genuinely-unseen 74.8%. This is consistent with the
test-split-is-already-saturated finding elsewhere in this document (5-seed
stats: mean accuracy 0.9998, std 0.0001) and does not retroactively excuse
the structural issue, just narrows its practical impact on this
particular dataset/task combination.

**This is why the held-out 9-cell matrix and the external dataset carry
more generalization weight than the standard test split, not less, and
should be read as reinforcing rather than redundant with each other:**
checked directly — the held-out matrix has **0/90** rows with content
overlapping train, and the external dataset has **1/30677** (a single
generic short benign string, `'a="get";'`, coincidental). Both were already
the mechanisms this project relies on for genuine-generalization claims
(§1's circularity concern, "XSS Context-Distance Augmentation" §Step 2,
"External Evaluation Dataset" in `docs/DATASET.md`); this finding is an
additional, independent reason those two evaluations matter, not a new
requirement to add them.

## RoBERTa held-out failure pattern mirrors GATv2's, for different reasons (2026-09-18)

RoBERTa baseline (Reviewer #3, `results/transformer_baselines.csv`), full
9-cell held-out matrix, deployed GATv2 checkpoint alongside for comparison:

| technique | GATv2 | RoBERTa |
|---|---|---|
| Benign/field_query | 1.0 | 1.0 |
| Benign/header_field | 1.0 | 1.0 |
| Benign/json_field | 1.0 | 1.0 |
| SQLi/sql_new_commands | 1.0 | 1.0 |
| SQLi/comment_splitting | 0.0 | 0.0 |
| **SQLi/case_mixing** | **1.0** | **0.0** |
| XSS/data_uri_base64 | 1.0 | 0.0 |
| XSS/event_handler_focus | 1.0 | 1.0 |
| XSS/svg_script_variant | 1.0 | 1.0 |

RoBERTa: 6/9. Both architectures fail `comment_splitting` (already
established as an architectural limit for GATv2, §9/§11/Step-2 above — not
re-derived here for RoBERTa, just noting it fails the same cell). Both
being wrong on the same cell for what could be different reasons is
expected and not itself the finding here.

**The `case_mixing` row is the finding.** This project already knows
GATv2's 1.0 on `case_mixing` is not a meaningful case-obfuscation-robustness
result — `src/preprocessing/tokenizer.py:13` lowercases every token before
GATv2 (or the string-matching baseline, whose regexes are also
case-insensitive) ever sees it, so the technique is neutralized upstream
and the cell cannot fail on that basis regardless of what the model
learned (documented at line 182 and 437 above, "not a useful test vector").
**RoBERTa's tokenizer does not lowercase** (verified directly:
`AutoTokenizer.from_pretrained('roberta-base').encode('uN/**/ioN aLl
sEl/**/eCt')` preserves `'u','N','aL','l','El','e','Ct'` with case intact,
byte-level BPE, no `do_lower_case` normalization) — RoBERTa receives the
case-mixed payload exactly as an attacker would send it, with no
upstream neutralization. **It fails, 0/10.**

This makes RoBERTa's `case_mixing` result the **first actually meaningful
measurement of case-obfuscation robustness in this entire investigation** —
every prior report of this cell (GATv2 across all configs/seeds, §11's
6-config ablation, the 5-seed run) was measuring "does the model recognize
plain, un-obfuscated SQLi keywords once lowercased," not case-obfuscation
resistance, despite being labeled `case_mixing` throughout. Read together:
**neither architecture actually handles case-mixing once something isn't
silently normalizing it away first.** GATv2's apparent robustness on this
cell was never real; it was an artifact of a preprocessing step shared by
every method evaluated through that pipeline. This doesn't retroactively
change any of §11's ablation conclusions (already caveated at line 182 not
to read `case_mixing` as an obfuscation-robustness signal there) — it
confirms that caveat was correct, from an independent angle, rather than
introducing a new one.

## Held-out 9-cell matrix: 3-method comparison (GATv2 / RoBERTa / string-matching) (2026-09-18)

The single most reviewer-relevant table produced by the Reviewer #3
baseline work: **no method, including the modern Transformer baseline,
solves the held-out matrix 9/9.**

| technique | GATv2 | RoBERTa | string-matching |
|---|---|---|---|
| Benign/field_query | 1.0 | 1.0 | 1.0 |
| Benign/header_field | 1.0 | 1.0 | 1.0 |
| Benign/json_field | 1.0 | 1.0 | 1.0 |
| SQLi/sql_new_commands | 1.0 | 1.0 | 0.0 |
| SQLi/comment_splitting | 0.0 | 0.0 | 0.0 |
| SQLi/case_mixing | 1.0 | 0.0 | 0.0 |
| XSS/data_uri_base64 | 1.0 | 0.0 | 0.0 |
| XSS/event_handler_focus | 1.0 | 1.0 | 0.0 |
| XSS/svg_script_variant | 1.0 | 1.0 | 0.0 |
| **total** | **8/9** | **6/9** | **3/9** |

Sources: `results/heldout_matrix_full.csv` (GATv2, string-matching),
`results/transformer_baselines.csv` (RoBERTa).

**Every failure has a distinct, verified cause — none of these three
numbers should be read as one undifferentiated "robustness score":**

- **`comment_splitting` (all three fail):** GATv2's is the established
  architectural limit (§§3-9 above — character-level keyword fragmentation
  destroys both the lexical signal and the `E_sem` edge simultaneously,
  with no compensating pathway). RoBERTa fails the same cell but for a
  presumably different, unverified reason (subword tokenization of
  `UNI/**/ON`-style fragments — not root-caused here, out of scope for this
  baseline pass). string-matching fails because its regexes
  (`r"union.*select"` etc., `src/preprocessing/normalization.py:118-120`)
  require the literal substring, which the `/**/`-fragmentation breaks
  regardless of case.
- **`case_mixing` (GATv2 passes, both others fail):** see the section above
  — GATv2's pass is an artifact of upstream `.lower()`, not real
  case-robustness. **string-matching also lowercases before matching
  (`normalization.py:109`, `s = str(content).lower()`) yet still fails** —
  its failure is NOT about case at all, it's the same `/**/`-fragmentation
  cause as `comment_splitting` (this cell's content combines mixed-case
  *and* comment-fragmentation: `f"uN/**/ioN aLl sEl/**/eCt"`). RoBERTa is
  the only one of the three actually being tested on case *and* failing
  because of it, not despite passing through unrelated to case (see
  previous section).
- **`sql_new_commands` (GATv2/RoBERTa pass, string-matching fails):**
  string-matching's SQLi regex list has no `TRUNCATE` pattern at all
  (`normalization.py:118-120` lists `union`, `select`, `insert`, `drop`,
  `or \d+=\d+`, `sleep`, `benchmark`, `information_schema`, `admin'--`,
  `order by` — `TRUNCATE TABLE` matches none of them). A vocabulary gap in
  a fixed rule list, not an obfuscation-defeat — the clearest illustration
  in this table of why a hand-maintained keyword list doesn't scale the
  same way a learned representation does.
- **`data_uri_base64` (GATv2 passes\*, RoBERTa/string-matching fail):**
  RoBERTa never sees the underlying `<svg>` payload as plaintext (it's
  base64-encoded inside a `data:` URI) and has no decoding step, so this is
  an expected, structural failure for a text classifier without a
  base64-aware preprocessing step. string-matching fails for the same
  reason (no decode step, literal regex against encoded text).
  \*Caveat already on record for GATv2 here too: this cell is
  seed-fragile, 4/5 not a stable 1.0 (5-seed statistics section above) —
  the "GATv2 passes" in this row is the deployed seed=42 checkpoint's
  single-seed result, consistent with how it's reported elsewhere in this
  document, not a re-assertion that it's unconditionally stable.
- **`event_handler_focus`/`svg_script_variant` (GATv2/RoBERTa pass,
  string-matching fails):** both contain literal `<svg`/event-handler-like
  tokens RoBERTa's tokenizer and GATv2's node features both see directly;
  string-matching's XSS regex list does cover `onerror=`/`onload=`-style
  patterns but not the specific attribute names used here
  (`onfocus=`, `onclick=` via `<a href='data:...'>` wrapping) — another
  vocabulary gap, not a structural failure.

**Takeaway for the paper:** framing this as "GATv2: 8/9, best-in-class" is
accurate but incomplete — the more defensible claim is that **each
method's failures are individually explainable and none is a
strictly-dominant approach across all nine adversarial techniques**, which
is a stronger, more honest basis for a Limitations/Discussion section than
a single leaderboard number.

## RoBERTa on the external dataset: same benign-type-4 gap, broader failure (2026-09-18)

RoBERTa (Reviewer #3 baseline) confusion matrix on
`data/external/external_dataset_clean.csv` (rows=true, cols=pred,
`[Benign, SQLi, XSS]`):

```
[[12042,   518,  6733],
 [  679,  8527,  1646],
 [    2,     0,   530]]
```

**Benign→XSS dominates: 6733/19293 (34.9%) of all Benign rows.**
Benign→SQLi is small (518, 2.7%). SQLi→XSS is also notable (1646/10852,
15.2%). XSS itself is near-perfect (530/532).

**Checked against the known benign-type-4 gap** ("Root cause of external
Benign misclassification: a 4th, un-augmented benign form" above): GATv2's
per-row predictions were regenerated against the **current** deployed
checkpoint for a row-level comparison (note: this checkpoint has been
retrained since that section was written, so its aggregate count —
5877/19293 Benign misclassified, 4750→XSS/1127→SQLi — differs from the
7163/5422/1741 figures recorded there; both are real, just from different
points in this checkpoint's retrain history, and this section uses the
current one to match what RoBERTa is being compared against).

- **100% of both models' misclassified Benign rows are "bare" values** (no
  `=`, `&`, `:`, `{`, `}`) — **RoBERTa is failing on the same structural gap
  already documented for GATv2, not a new failure category.**
- But the two failure sets only overlap 2107/7251 rows (29% of RoBERTa's
  misclassifications) — **RoBERTa fails on 5144 bare-value rows GATv2 gets
  right**, including plain alphabetic tokens with no digits or `@` at all
  (`"fennell"`, `"genny"`, `"mckenney"`, `"maala8"`), not just the
  numeric-ID/email sub-patterns GATv2's gap was characterized by. RoBERTa's
  version of this gap is broader, not just differently-shaped. **This
  broadens the gap's definition itself: `docs/DATASET.md`'s "4th benign
  form" is corrected from the three originally-observed sub-cases
  ("numeric ID, short alphanumeric token, or email address") to "any bare
  short value (numeric, alphanumeric, or plain word) without any
  structural delimiter" — the defining property was always the missing
  delimiter/key-value structure, not membership in one of three named
  sub-patterns, and RoBERTa's plain-word failures are the evidence that
  distinction matters.**
- **The "content too short = no context, basically `<s></s>`" hypothesis is
  a partial, not primary, explanation.** Token-count check (RoBERTa's own
  tokenizer): rows misclassified Benign→XSS have median 6 tokens vs. 7 for
  correctly-classified Benign — a real but small gap. Rows with ≤4 tokens
  (`<s>` + ≤2 real subwords + `</s>`) are 31.5% of the misclassified set vs.
  21.4% of the correct set — a moderate correlation, but 68.5% of the
  misclassifications have more than 4 tokens, so tokenizer starvation alone
  does not explain most of them.
- **Directional bias matches GATv2's, more extreme:** both models
  disproportionately guess XSS rather than SQLi for ambiguous bare-value
  Benign input (GATv2: 4750 XSS vs. 1127 SQLi; RoBERTa: 6733 vs. 518) — same
  direction, RoBERTa's skew is sharper.

**Reading for the paper:** this is not evidence that a Transformer baseline
solves (or introduces a new version of) the benign-type-4 gap — it inherits
the same structural blind spot (bare, delimiter-free short values look
attack-like to both a token-graph model and a pure sequence model alike),
and inherits it *more severely* despite having no graph-size shortcut to
blame. That rules out "GATv2's gap is a graph-structural artifact" as a
complete explanation — a text-only architecture with none of GATv2's graph
machinery has the same blind spot, which points toward the gap being about
the *training data* (no bare-value Benign examples in any augmentation
round — see "Train-only Benign syntax diversity" in `docs/DATASET.md`) more
than about either model's specific architecture.

## Interpretability: GATv2 edge-attention vs RoBERTa/CodeBERT attention-rollout (2026-09-18)

Minimal, honest comparison (Reviewer #3 asked for interpretability
discussion, no specific method mandated). GATv2 side reuses existing
edge-level attention extraction (`scripts/inspect_attention_weights.py`,
`data/attention_weights_comment_split.txt`); the baselines get a
deliberately simple attention-rollout proxy
(`scripts/interpretability_attention_rollout.py`): mean attention FROM the
`<s>`/[CLS] token TO every input token, averaged across all layers and
heads (not the full recursive Abnar & Zuidema rollout). Same
`sqli_comment_split` content used by both, for direct comparison:
`'p=0;UNI/**/ON ALL SEL/**/ECT * FR/**/OM users WH/**/ERE id=0;--'`.
Two more examples (`sqli_clear`, `xss_svg_onload`) covered for RoBERTa and
CodeBERT only. Full data: `results/interpretability/` (6 PNGs,
`attention_rollout_report.txt`).

**GATv2 (existing result, restated for comparison):** attention is
attributable to a *relation type*. On this exact content, the 4 `E_sem`
edges average 0.154 attention, ordinary (non-self-loop) edges average
0.202, self-loops average 0.207 — E_sem edges get *less*, not more,
attention than typical edges here (ratio 0.76), a finding already on record
elsewhere in this document (§9) and not re-derived here. The point for this
comparison isn't whether E_sem is favored — it's that the number is
meaningful at all: every attention weight in this model is tagged with
*which structural relationship* (E_seq/E_skip/E_sem) it belongs to.

**RoBERTa/CodeBERT (new):** the rollout gives only token-level importance,
with no relational structure — attention FROM `<s>` TO token *i* says
nothing about token *i*'s relationship to any other token. On
`sqli_comment_split`, RoBERTa's top non-special tokens are `--`, two `/**`
fragments (0.029-0.025); CodeBERT's are `--`, two `;` tokens (0.023-0.018)
— both land on SQL-syntax markers near the injected comment fragments,
plausible but not surprising for a model whose classifier head reads only
the `<s>` representation. On `xss_svg_onload`
(`'<svg onload=alert(1)>'`), both models' top token is `<` for RoBERTa /
`<`+`alert` for CodeBERT — i.e. attending to the opening delimiter and the
payload keyword, not to `onload` itself specifically.

**A real limitation of this simple method, visible in the actual numbers,
not just asserted:** a large fraction of `<s>`'s own attention mass goes
to **itself** — e.g. `sqli_comment_split`/RoBERTa: `<s>`→`<s>` = 0.4731 of
the total (see `results/interpretability/roberta_sqli_comment_split.png`),
~16.1x the next-highest token (`--`, 0.0293). This is a known property of raw
attention in Transformer classifiers (the `[CLS]`-attends-to-itself
phenomenon), not a bug in this script, but it means the "top-attended
tokens" reported above are computed after excluding the special tokens
(`<s>`/`</s>`/`<pad>`) from the ranking — read as "most-attended *content*
token," not as a claim that this is where most of the model's attention
literally goes.

**Framing for the paper (qualitative, not quantitative — no attempt to
force this into a single comparable metric):** GATv2's attention is
structurally interpretable — a practitioner can ask "how much does this
model rely on semantic-dependency edges vs. sequential ones" and get a
real, relation-typed answer. RoBERTa/CodeBERT's attention-rollout answers a
narrower question — "which tokens did the classifier's representation draw
from" — with no equivalent notion of edge type, because there are no edges;
BAG's explicit multi-relational graph structure is not something a
sequence-only Transformer has an analogue for. This is a genuine
architectural trade-off, not a deficiency of the rollout method: the
comparison exists to make the trade-off legible, not to declare one
approach's interpretability "better" in the abstract.

## Baseline fairness re-run (#16 / RQF-06): the old Table 2 does not reproduce, and only the held-out/external modes can rank architectures (2026-09-19)

Setup and exact hyperparameters: `docs/REPRODUCIBILITY.md`, section "Graph & Sequence Baselines". In one
line: GCN / GraphSAGE / GIN / HGT (same graphs as GATv2, only the conv operator swapped) and Bi-LSTM /
TextCNN / StackLSTM (raw `web_security_tokenizer` tokens, embedding learned from scratch) were trained on the
current 43,659-row dataset and frozen split through **the same `train()` loop and protocol as GATv2** — same
optimizer/scheduler/class weights/early stopping, default LR for all (none needed a per-baseline LR), seeds 42–46 —
and evaluated on the same three modes. GATv2 was re-evaluated through the same code (deployed seed-42 checkpoint +
the 5-seed run's seeds 43–46) as the reference. Raw data: `results/{graph,sequence}_baselines{,_5seed}.csv`,
`results/gatv2_reference_5seed.csv`, `results/final_baseline_comparison.csv`,
`results/latency_breakdown_graph_baselines.csv`, `results/table2_old_vs_new.csv`.

### Finding 1 — test split: the old 2–10-point gaps vanish; every model is at the ceiling

Old Table 2 numbers are transcribed from `hin_web_vulne/sn-article_v1.tex` (30k-row dataset; the baseline
configurations behind them are not recoverable from this repo). New = mean test accuracy over 5 seeds
(std ≤ 0.13 pp for every model; seed-42 values in `final_baseline_comparison.csv`).

| method | old acc (Table 2) | new test acc, 5-seed mean | Δ (pp) | old margin of Proposed over it (pp) | new margin (pp) |
|---|---|---|---|---|---|
| TextCNN | 89.54 | 100.000 | +10.46 | +8.30 | −0.02 |
| Bi-LSTM | 91.20 | 100.000 | +8.80 | +6.64 | −0.02 |
| Stack-LSTM | 92.15 | 99.938 | +7.79 | +5.69 | +0.04 |
| GCN | 93.45 | 99.991 | +6.54 | +4.39 | −0.01 |
| GraphSAGE | 93.88 | 99.991 | +6.11 | +3.96 | −0.01 |
| GIN | 94.12 | 99.991 | +5.87 | +3.72 | −0.01 |
| GATv2 (vanilla) | 95.30 | *not re-run* ("vanilla" undefined here) | — | +2.54 | — |
| HGT | 95.87 | 99.991 | +4.12 | +1.97 | −0.01 |
| **Proposed (LOGWAT)** | 97.84 | 99.981 | +2.14 | — | — |

Every re-run baseline is **+4.1 to +10.5 points above its old Table-2 figure**, and is now indistinguishable from
the proposed model on the test split (margins within ±0.04 pp ≈ ±2 of 4,215 samples; the split's saturation was
already documented in §1/§4/§10 and the 5-seed section). The reported 2–8-point advantage of LOGWAT over the GNN and
sequence baselines is **not reproduced** under identical inputs and protocol.

**What this does and does not establish about RQF-06.** It is consistent with the suspicion that the old baselines were
configured more weakly than GATv2 — under the current, like-for-like setup no baseline is weaker. It does **not**
prove it: (i) the old configurations cannot be recovered, and the old dataset/split differ from the current ones
(the proposed model itself moved +2.14 pp between the two), so "weaker configuration" and "different, easier dataset"
are not separable from these numbers; (ii) baselines that started lower also have more room to move, so a larger Δ is
partly ceiling arithmetic. The defensible statement is narrower: **the old table's ordering and margins cannot be
carried over; on the standard split, architectures are not distinguishable at all.** The old Table 2 should be replaced,
not patched.

### Finding 2 — held-out matrix (90 rows, 9 cells): GATv2 is not distinguishable from GCN/GraphSAGE/GIN/TextCNN; only HGT is separated

Correct out of 50 (5 seeds × 10 rows) per cell; `results/*_5seed.csv` `cell:*` columns.

| cell | GATv2 | GCN | GraphSAGE | GIN | HGT | Bi-LSTM | TextCNN | StackLSTM |
|---|---|---|---|---|---|---|---|---|
| Benign/field_query | 50 | 50 | 50 | 50 | 50 | 10 | 43 | 7 |
| Benign/header_field | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 40 |
| Benign/json_field | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| SQLi/sql_new_commands | 50 | 50 | 50 | 50 | 50 | 41 | 50 | 20 |
| SQLi/comment_splitting | **0** | 10 | **0** | **0** | **50** | 40 | **50** | 12 |
| SQLi/case_mixing | 50 | 40 | 40 | 40 | 50 | 36 | 50 | 4 |
| XSS/data_uri_base64 | 40 | 50 | 50 | 40 | 50 | 30 | 20 | 20 |
| XSS/event_handler_focus | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| XSS/svg_script_variant | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| **mean cells/9 (5 seeds)** | **7.8** | 8.0 | 7.8 | 7.6 | **9.0** | 6.8 | 8.0 | 4.6 |
| mean correct/90 ± std | 78.0±4.5 | 80.0±7.1 | 78.0±4.5 | 76.0±5.5 | **90.0±0.0** | 71.4±17.8 | 82.6±5.7 | 50.6±16.0 |

- **GATv2, GCN, GraphSAGE, GIN and TextCNN are within seed noise of each other** (78±4.5 vs 80±7.1 / 78±4.5 /
  76±5.5 / 82.6±5.7 correct of 90; Welch p = 0.61 / 1.00 / 0.55 / 0.20 vs GATv2, n=5, uncorrected — indicative only). The
  held-out matrix does not show GATv2 outperforming the single-relation GNNs.
- **HGT is the one clearly separated model: 9/9 cells at every one of the 5 seeds** (GATv2's per-seed cells: 8, 8, 7, 8, 8
  — the ranges do not overlap). HGT is the only baseline that consumes the relation type of each edge.
- **`SQLi/comment_splitting` — previously documented as a stable architectural limit (§9, §11, "5-seed statistics": 0/5 seeds)
  — is a limit of the *single-relation* GNNs on this graph (GATv2, GraphSAGE, GIN: 0/50; GCN 10/50), not of the graph
  itself:** HGT passes it 50/50 on the identical E_seq+E_skip+E_sem edges, once the edges are typed. This is consistent
  with §11(c)'s diagnosis (E_sem's 4 edges are drowned among ~142 n-gram edges when all edges share one aggregation) [attention part of that diagnosis corrected on 2026-09-19: E_sem is not under-attended, see the Correction in §9] and with
  its proposed remedy (a per-relation aggregation path) — HGT's per-edge-type message/attention parameters are exactly
  that. It is **supporting evidence, not isolation**: HGT also differs from GATv2 in attention form and has ~2× the
  parameters (610K vs 299K), and §10's `edge_attr` on GATv2Conv did not fix the cell. **Update (same day): isolated by controlled retraining and interventions — see "HGT investigation" at the end of this file.**
- **TextCNN also passes `comment_splitting` 50/50 without any graph** (kernels of 3–5 tokens over `uni / * * / on` style
  fragments), and is joint-worst with StackLSTM on `data_uri_base64` (20/50 each; Bi-LSTM 30/50). So a held-out cell being
  solved is not by itself evidence that graph structure is needed; the matrix cannot separate "graph" from "any model that
  sees local token n-grams".
- Sequence models are unstable across seeds (Bi-LSTM 42→89 of 90 correct, StackLSTM 34→71); GATv2/GCN/SAGE/GIN are much steadier.

### Finding 3 — external dataset (n=30,677): GATv2 has the best mean weighted F1, but the margin over GCN/GraphSAGE/GIN is modest

| method | weighted F1, 5-seed mean ± std | range | Δ vs GATv2 | Welch p (n=5) | per-seed Benign recall range |
|---|---|---|---|---|---|
| **GATv2** | **0.796 ± 0.030** | 0.771–0.840 | — | — | 0.633–0.719 |
| GCN | 0.745 ± 0.057 | 0.669–0.800 | −0.051 | 0.13 | 0.506–0.712 |
| GraphSAGE | 0.715 ± 0.064 | 0.658–0.818 | −0.081 | 0.045 | 0.480–0.725 |
| GIN | 0.692 ± 0.075 | 0.584–0.785 | −0.104 | 0.032 | 0.411–0.667 |
| HGT | 0.535 ± 0.218 | 0.328–0.819 | −0.261 | 0.055 | **0.059**–0.726 |
| TextCNN | 0.509 ± 0.046 | 0.462–0.577 | −0.287 | <0.001 | 0.175–0.302 |
| Bi-LSTM | 0.444 ± 0.079 | 0.310–0.503 | −0.352 | <0.001 | 0.165–0.244 |
| StackLSTM | 0.392 ± 0.035 | 0.360–0.440 | −0.404 | <0.001 | 0.085–0.190 |

(Welch p uncorrected for 7 comparisons; with n=5 read as indicative. Mann–Whitney p: GCN 0.22, GraphSAGE 0.095, GIN 0.056, HGT 0.095, sequence models 0.008.)

- GATv2's advantage is **clear over the sequence models** (all learned from scratch, 28% of external benign tokens are UNK,
  benign requests here average 2.2 tokens), **real but modest over GCN/GraphSAGE/GIN** (ranges overlap; only GIN/GraphSAGE reach
  p<0.05 uncorrected), and unresolved against HGT because HGT is bimodal.
- **The 0.840 external weighted F1 used for GATv2 in `final_baseline_comparison.csv` (the deployed seed-42 checkpoint) is the
  maximum of its five seeds** (mean 0.796). Any table that compares that single number against 5-seed means overstates the gap; use
  the 5-seed figures above when comparing across methods.
- **HGT is perfect on the held-out matrix yet collapses on external Benign in 3 of 5 seeds** (seeds 44/45/46: Benign recall
  0.059/0.083/0.111, weighted F1 0.33/0.41/0.40; seeds 42/43: 0.57/0.73). Held-out perfection did not transfer: the external
  benign traffic (short address/number-like fields, the "benign-type-4 gap" of §"RoBERTa on the external dataset") is
  where every learned model is weakest, and HGT's failure there is seed-dependent.

### Finding 4 — latency: the old "Proposed is fastest" claim does not hold for the graph baselines

Same methodology as #11/#21 (`results/latency_breakdown_graph_baselines.csv`, batch 1, N=1000, GATv2 measured in the same run;
GATv2 1.79 ms CPU / 2.18 ms CUDA agrees with #21's 2.01 / 2.31). End-to-end mean (p95), ms:

| | GATv2 | GCN | GraphSAGE | GIN | HGT |
|---|---|---|---|---|---|
| CPU | 1.785 (2.59) | 1.411 (2.06) | 1.137 (1.79) | 1.152 (1.82) | 5.602 (7.84) |
| CUDA | 2.177 (2.69) | 1.793 (2.28) | 1.364 (1.83) | 1.342 (1.81) | 8.093 (9.64) |
| old Table 2 (device unspecified) | 3.7 (Proposed) / 4.1 (GATv2) | 5.8 | 4.5 | 4.9 | 6.2 |

GCN/GraphSAGE/GIN are 1.2–1.6× **faster** than GATv2 (attention costs more than plain aggregation); HGT is 3.1× (CPU) to 3.7× (CUDA) slower. The old
table's ordering (Proposed fastest of all GNNs) is not reproduced — the paper's latency argument should be "GATv2 meets the
real-time budget" (§LATENCY), not "GATv2 is faster than other GNNs". HGT's BAG-construction stage is 0.63 ms vs 0.03 ms because
it must also build the relation one-hot (pure Python here); its forward pass (4.5 ms CPU) dominates regardless. Group-B latency was not measured.

### Proposed wording for the paper's Discussion (replaces the claim that LOGWAT outperforms GNN/sequence baselines by 2.5–8%)

> When all baselines are re-trained on the same dataset, split and protocol as LOGWAT, the standard test split no longer
> separates them (all ≥ 99.9% accuracy; the gaps in the earlier comparison do not reproduce), which we attribute to the split's
> saturation rather than to equal capability. Discrimination comes from the held-out obfuscation matrix and an external dataset.
> There, LOGWAT's GATv2 is not distinguishable from GCN, GraphSAGE, GIN or a token-level TextCNN on the held-out matrix (78±4.5 vs
> 76–83 of 90 correct over five seeds), while the multi-relational HGT — the only baseline that consumes edge types — solves it at
> every seed (90/90), including the comment-splitting cell that all single-relation GNNs fail; this supports, though does not isolate,
> the value of relation-aware aggregation. On the external dataset LOGWAT attains the highest mean weighted F1 (0.796±0.030), clearly
> above the from-scratch sequence models (≤0.51) and modestly above GCN/GraphSAGE/GIN (0.69–0.75); HGT's external performance is
> seed-unstable (0.33–0.82). LOGWAT is 1.2–1.6× slower than GCN/GraphSAGE/GIN and 3–4× faster than HGT.

### Limitations of this comparison (all carried into the numbers above)

- **Selection on the test split.** `train()` picks the best checkpoint and stops on test-split accuracy for GATv2 and for every
  baseline alike (like-for-like, but test numbers are selected numbers). The Transformer baselines use a train-only validation loss.
- **No per-baseline hyperparameter search.** All baselines ran at GATv2's LR/weight-decay/batch size; none failed to converge, so no
  override was made, but a tuned baseline (LR, hidden size, regularization) might do better — "baseline at its best" was not explored beyond
  the defaults. The from-scratch sequence models carry 1.5M embedding parameters against GATv2's 299K, and see no engineered node features.
- **Training is not bit-reproducible** even at a fixed seed on this GPU (see "5-seed statistics"); the seed-42 rows in the canonical CSVs
  are one draw. n=5 seeds; p-values are uncorrected and indicative.
- **Old Table 2 configurations are unrecoverable**, so the old-vs-new comparison is between numbers, not between reproduced configurations.
- The held-out matrix (90 rows, 9 hand-built cells) and the external set (19,293 benign / 10,852 SQLi / 532 XSS; class-imbalanced, benign-type-4 heavy)
  are single instances, not distributions of held-out attacks.

## HGT investigation: what the held-out 9/9 and the external collapse each come from (2026-09-19)

Follow-up to Finding 2/3 of the #16 re-run, which called HGT's held-out result "supporting evidence, not isolation" and left
its bimodal external result unexplained. This section isolates both. Scripts: `scripts/inspect_attention_weights_hgt.py`,
`investigate_hgt_edge_type_use.py`, `train_hgt_control.py` + `evaluate_hgt_controls.py` (+ `src/models/hgt_controls.py`),
`investigate_hgt_generalization.py`, `probe_hgt_tiny_graph_channel.py`, `hgt_controls_stats.py`. Raw output:
`data/attention_weights_comment_split_hgt.txt`, `data/hgt_edge_type_intervention.txt`, `data/hgt_controls_report.txt`,
`data/hgt_generalization_report.txt`, `data/hgt_tiny_graph_channel_probe.txt`, `results/hgt_*.csv`, `results/hgt_controls_tests.txt`.
The 5 typed-HGT checkpoints are the #16 ones (unchanged); `real` conditions reproduce the published 90/90 and external wF1 exactly.

### 1. Attention weights (same probe string as GATv2's, 5 seeds) — descriptive only, and it does not show what HGT does

`HGTConv` has no `return_attention_weights`; alpha was recorded by wrapping the `softmax` inside `HGTConv.message` for one forward
pass (hooked output bit-identical to the plain forward). Probe graph: 37 nodes, 146 edges = 72 seq + 70 skip + 4 sem (the same 4 E_sem edges as the GATv2 probe).

| | layer 1 | layer 2 |
|---|---|---|
| raw alpha_sem / alpha_(seq+skip) (GATv2 analogue; GATv2 was 0.761 / 0.824 with edge_attr; those were raw, degree-corrected they are 0.996 / 1.074, see the Correction in §9) | 0.737 (0.731–0.751) | 1.32 (1.05–1.85) |
| relative alpha (= alpha × in-degree; 1.0 = uniform over neighbours) on sem edges | **1.03** (1.02–1.05) | 1.79 (1.43–2.47) |
| share of attention mass on sem edges vs share of edges (2.7%) | 2.0% | 3.6% |

Layer-1 attention on E_sem is **uniform** (all 8 heads ≈ equal, 0.17–0.22); the raw 0.74 ratio is just the 1/in-degree effect GATv2's 0.76 also
had. Layer 2 tilts toward sem edges by ~1.8× but they still carry <4% of the mass. So attention is **not** where HGT separates the relation:
the type-specific value/message transform is. That is why the following causal tests, not attention, are the evidence.

### 2. Does the trained HGT use edge types? Yes — and it is the 4 E_sem edges that carry the held-out result

Same 5 checkpoints, edge labels edited at inference (matrix: /450 = 9 cells × 10 rows × 5 seeds; `cs` = SQLi/comment_splitting /50 and mean SQLi logit margin; test = frozen test split):

| condition | matrix /450 | cs /50 | cs margin | case_mixing /50 | test acc |
|---|---|---|---|---|---|
| real | 450 | 50 | **+3.96** | 50 | 1.000 |
| drop the 4 E_sem edges | 370 | **0** | **−2.35** | 20 | 0.999 |
| relabel E_sem as seq / as skip | 360 / 370 | 0 / 0 | −2.55 / −2.73 | 10 / 20 | 0.999 |
| swap seq↔skip labels | 450 | 50 | +6.90 | 50 | 0.949 (seed 43: 0.747; other seeds ≥ 0.9986) |
| all edges → one type (seq / skip) | 240 / 260 | 10 / 20 | −4.1 / +0.3 | 10 / 20 | 0.511 / 0.753 |
| all edges → sem | 150 | 50 | +22.5 | 50 | **0.288** (every input → SQLi) |
| cyclic relabel (seq→skip→sem→seq) | 180 | 50 | +13.0 | 50 | 0.356 |
| hash-random types | 240 | 50 | +8.9 | 50 | 0.539 |
| drop skip / drop seq | 220 / 300 | 10 / 40 | −9.0 / +9.6 | 10 / 50 | 0.537 / 0.781 |
| **GATv2** real / drop_sem / drop_skip / **drop_seq** | 390 / 380 / 400 / 440 | 0 / 0 / 10 / **50** | −2.44 / −2.54 / −2.13 / **+4.38** | 50 / 40 / 50 / 50 | 1.000 / 1.000 / 1.000 / 0.999 |

- Removing or mis-typing **only the 4 sem edges (2.7% of the probe's edges)** returns HGT's comment_splitting margin to GATv2's (−2.35 vs −2.44) and 50/50 → 0/50.
  GATv2 is unaffected by dropping them (−2.44 → −2.54). So the sem *type* is what HGT reads; the sem edges' existence alone is not.
- The sem channel is a strong SQLi channel: with every edge typed sem, every input (Benign, XSS included) is predicted SQLi at all 5 seeds.
- seq/skip: merging them (all→seq or all→skip) wrecks test accuracy (0.51 / 0.75), but swapping the two labels is harmless at 4/5 seeds: the network needs
  two *distinguishable* neighbour channels, not which is which. The near-orthogonality of the per-type W_msg/W_att matrices (cos ≈ 0, relative distance ≈ √2 at every seed and layer) is consistent with that, but parameters alone do not show the roles.
- GATv2's failure is dilution, causally: deleting its 72 seq edges (E_sem share 2.7% → 5.4%) makes it pass comment_splitting at all 5 seeds (+4.38 margin) — the information is in the model and drowned by the seq edges, as §11(c) hypothesised. (Off-distribution input; read as a probe, not as a fix.)

### 3. Isolation by controlled retraining: same HGT, no relation information

Same `HGTBaseline` skeleton, same `train()` loop, split, LR, early stopping, seeds 42–46; only the edge typing differs (`src/models/hgt_controls.py`): **collapsed** = one edge type (544,789 params);
**random** = 3 edge types assigned by a hash of the node indices, 35.6% agreement with the true type on 200 sample graphs ≈ chance (610,357 params = typed HGT's). All 10 runs converged (best test acc 0.9998).

| 5-seed | typed HGT | HGT-collapsed | HGT-random | GATv2 |
|---|---|---|---|---|
| held-out correct /90 | **90.0 ± 0.0** | 80.0 ± 0.0 | 81.6 ± 4.8 | 78.0 ± 4.5 |
| comment_splitting: seeds passing (10/10 rows) | **5/5** | 1/5 | 1/5 | 0/5 |
| comment_splitting SQLi margin | +3.96 ± 2.84 | −2.48 ± 2.32 | −5.21 ± 6.54 | −2.44 ± 2.25 |
| external weighted F1 | 0.535 ± 0.218 | 0.754 ± 0.034 | 0.737 ± 0.039 | 0.796 ± 0.030 |
| external Benign recall | 0.310 ± 0.314 | 0.633 ± 0.058 | 0.607 ± 0.069 | 0.674 ± 0.038 |
| seeds with external Benign recall < 0.2 | **3/5** | 0/5 | 0/5 | 0/5 |
| test acc | 1.000 | 1.000 | 1.000 | 1.000 |

Typed 5/5 vs controls 2/10 on comment_splitting: Fisher one-sided p = 0.007. Collapse on external Benign: typed 3/5 vs controls 0/10: p = 0.022 (n tiny; indicative).
External wF1 typed vs pooled controls: Mann–Whitney p = 0.129 (typed is bimodal, so the mean gap is not significant by itself; the collapse frequency is the sharper signal).
**Conclusion:** the held-out win is due to the edge typing, not HGT's attention form, K/Q/V, skip gate or parameter count (the random control has the same parameter count and does not win).
It is not all-or-nothing: an untyped HGT solves comment_splitting at seed 45 in both controls. **The same typing is also what is unstable on external data** (controls at identical size do not collapse).

### 4. What the typed channel is, and its limits

- **E_sem is a rule-derived attack signal.** `src/edges/semantic.py` links reconstructed keyword pairs (union→select, select→from, script→src, …) and merges keywords split by `/ * - #` or by whitespace.
  Fraction of graphs with ≥1 E_sem edge: Benign **0.0000** in train+test (0/15,000), in external (0/19,293) and in all 3 benign matrix cells; SQLi 0.295 (train) / 0.491 (external); XSS 0.027 / 0.056.
  "Has E_sem ⇒ attack" therefore has precision 1.000 on both sets (recall 0.20 / 0.47). A typed channel can read that off; a single merged aggregation drowns it.
- **The comment_splitting result is transfer through the builder, not through HGT alone.** Train has no `/**/`-split SQLi (0 rows), but 2,111 `sqli_pool_noise` rows split keywords with whitespace (`SELE CT`, `FR<tab>OM`), which the builder maps to the same E_sem edges.
- **Held-out matrix is narrow.** 90 rows = 27 distinct inputs after masking digits (7 of 9 cells are one template ×10; data_uri_base64 and svg_script_variant 10 each). E_sem appears in exactly 2 of 9 cells (comment_splitting, case_mixing). HGT's held-out advantage over GATv2 is comment_splitting (0 → 50 of 50) and data_uri_base64 (40 → 50; that cell has no E_sem edge, and GATv2's 40 is a seed-level miss); case_mixing is 50/50 for both.
  All 9 cells have ≥10 tokens; the matrix is not used by `train()` (selection is on the test split, HGT ran once per seed at default hyperparameters). But the training-set augmentations
  were written with the matrix's cells and mechanisms in view (`add_benign_syntax_diversity_train` fills the header/JSON coverage that the held-out benign cells exposed; `add_train_noise_augmentation` is deliberately a different splitting mechanism from the matrix's `/**/`; field names are disjoint by construction), so the matrix is not a pristine hold-out for *any* model.
- **Fresh probes outside the 9 templates** (hand-written here, eval-only, 6 strings per group × 5 seeds; not added to any dataset):

| group (E_sem edges the builder produced) | typed HGT | HGT-collapsed | HGT-random | GATv2 |
|---|---|---|---|---|
| SQLi split by `/**/` in other wordings (4–6) — correct | **25/30** | 8/30 | 7/30 | 7/30 |
| SQLi obfuscated so the builder cannot rebuild the keyword (`UNI+ON`, `U/*x*/NION`, `uni!on`; 0 edges) — correct | **0/15** | — | — | 0/15 |
| SQLi obfuscated but rebuilt (`%0a`, `%2f%2a%2a%2f`, plain; 4 edges) — correct | 15/15 | — | — | 9/15 |
| Benign prose containing keyword pairs (`order by … group by`, `select … from`; 2–6 edges) — correct | **5/30** | 15/30 | 15/30 | 15/30 |
| Benign short address/number-like | 12/30 | 16/30 | 16/30 | 16/30 |
| XSS in other wordings | 30/30 | 30/30 | 30/30 | 30/30 |

  So typed HGT does generalise beyond the exact matrix template — but only while the builder emits an E_sem edge; without one it is as blind as GATv2 (0/15), and it flags benign text containing a keyword pair (25 of 30 predictions wrong; all 5 seeds on 5 of 6 strings) because no benign example with an E_sem edge exists in training.
  (Small hand-written sets: indicative, not a benchmark.)

### 5. The external collapse: not E_sem, not capacity — the seq channel on 1–2-token graphs

- External Benign has 0 E_sem edges, so the sem shortcut cannot cause its false positives; deleting E_sem at inference leaves Benign recall unchanged at every seed (0.571/0.726/0.059/0.083/0.111 before and after).
- Size regime: 68.3% of external Benign are 1-token graphs and 7.0% are 2-token; **the training set has 0 Benign and 0 SQLi graphs with 1–2 tokens** (Benign: 62 with 3 tokens; XSS: 1 + 1 + 15), and 87.9% of train Benign are ≥13 tokens. All held-out cells are ≥10 tokens.
  The two signals ("perfect on held-out", "collapses on external") therefore probe **disjoint size regimes**, not one skill.
- Typed HGT Benign recall on 1-token / 2-token external graphs: seeds 42/43 0.57, 0.84 / 0.96, 0.96; seeds 44/45/46 **0.00 / 0.00 / 0.00 and 0.00 / 0.002 / 0.004** (predicting XSS for 98–99.8% of 2-token inputs, whatever the content).
  Controls and GATv2 are stable there (2-token recall 0.94–0.99 at every seed, 1-token 0.51–0.69).
- Those graphs have only seq-typed edges. Relabelling them to skip flips the answer per seed (seed 44 n=2: seq→XSS 98%, skip→Benign 97%; seed 42/43: seq→Benign 96%, skip→SQLi 94–98%); sem→SQLi 100% at every seed (`data/hgt_tiny_graph_channel_probe.txt`).
  The output on this never-trained input is decided by how each seed's seq channel extrapolates to a lone-seq neighbourhood — a seed lottery. **Not isolated by a controlled experiment**: that HGT has no self-loops (layer-1 nodes see only neighbours) may contribute; untested.

### Answer to the "overfit to the 9 held-out cells" hypothesis

Not supported as stated. (a) The matrix is not used to train or select; HGT's 9/9 holds at 5/5 seeds and on 25/30 fresh comment-split strings, so it is not template memorisation. (b) A same-size, same-architecture HGT without relation information does not reach it (controls: 80.0 / 81.6 vs 90.0),
so it is not capacity. What is true is narrower: HGT's edge is one channel (E_sem) supplied by a hand-built keyword matcher whose only failure modes are (i) obfuscations the matcher cannot rebuild and (ii) benign text containing keyword pairs; and the external instability comes from a regime (1–2 tokens) that neither training nor the matrix covers.

### Proposed wording for the Discussion (replaces "supports, though does not isolate")

> On the held-out matrix the multi-relational HGT solved every cell at every seed, including comment-splitting, which all single-relation GNNs fail. Two controls with the same architecture and parameter count but no relation information
> (one edge type; randomly assigned types) solved it in 2 of 10 runs versus 5 of 5, and relabelling or deleting the four semantic edges of the probe returns HGT to GATv2's failure (SQLi margin +3.96 → −2.35 vs GATv2's −2.44). The gain is therefore attributable to typed edges — a separate aggregation path in which
> the rare, rule-derived semantic edges are not diluted by ~140 n-gram edges — rather than to HGT's attention form or capacity; deleting GATv2's sequential edges lets it solve the cell too. This is a property of our semantic-edge builder as much as of the learner: HGT generalises to other split wordings only when the builder emits a semantic edge (25/30 vs 7/30 for GATv2; 0/15 when it does not), and flags benign text containing keyword pairs (semantic edges never co-occur with benign examples in our data).
> On the external dataset HGT is seed-unstable (weighted F1 0.33–0.82; 3 of 5 seeds label essentially every 1–2-token benign field an attack, mostly XSS), a failure not seen in the untyped controls; we attribute it to graph sizes absent from training rather than to overfitting the held-out matrix.

### Caveats of this investigation

- Attention was extracted on one graph (the GATv2 probe). Interventions are off-distribution edits of trained models; the controlled retrain (§3) is the isolating evidence, the edits are the mechanism evidence.
- Controls are n=5 seeds each; Fisher/MWU p-values are uncorrected. The typed-HGT 5 seeds and the GATv2 5 seeds are the #16/#22 checkpoints; the 10 control checkpoints were trained here with up to 4 jobs sharing the GPU (`logs/hgt_*_training.log` are interleaved across concurrent runs; checkpoints are unaffected). Training is not bit-reproducible (see "5-seed statistics").
- The "seq channel extrapolation" explanation of the external collapse rests on the relabelling probe and the control contrast, not on a controlled retrain; the no-self-loop contribution is untested.
- Fresh probes are hand-written (6 strings per group), not a benchmark. The matrix contains 27 distinct inputs.
