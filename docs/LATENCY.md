# Latency Measurements

## Environment (measured 2026-09-18)

- CPU: Intel(R) Core(TM) i7-10875H CPU @ 2.30GHz (16 logical cores)
- GPU: NVIDIA GeForce RTX 2080 Super with Max-Q Design
- torch: 2.6.0+cu124
- torch_geometric: 2.8.0.post1
- CUDA (`torch.version.cuda`): 12.4
- Script: `scripts/benchmark_latency.py`, full pinned dependency list:
  [`requirements.txt`](../requirements.txt)

## Methodology

- **batch_size=1** throughout — simulates a single-request WAF deployment,
  *not* training-time batching. A batched-inference number would be
  materially lower per-request and is not what a WAF actually does.
- **N=1000** real request contents, sampled with replacement (seed=42) from
  the frozen test split's `content` column — reflects realistic
  content-length variance, not one fixed string repeated 1000 times.
- **20 warm-up iterations** (untimed) before the 1000 measured ones, on each
  device — standard practice to exclude first-call/CUDA-context-init
  overhead from the measured distribution.
- **Model load time measured once, separately** (`HeavyWebGNN(...)` +
  `load_state_dict` + `.to(device)` + `.eval()`), **not** included in any
  per-request number below — a WAF loads the model once at startup, not per
  request.
- 4 stages timed via the *exact production code path* (not a
  reimplementation): `web_security_tokenizer`/`get_node_features` (stage 1),
  `sequential_edges`/`skip_edges`/`semantic_edges` (stage 2), the
  `HeavyWebGNN` forward pass including host→device transfer for the CUDA run
  (stage 3), and end-to-end wall-clock (stage 4, one continuous span per
  request, not just the sum of the other 3 stages' medians).
- `torch.cuda.synchronize()` called around GPU timing boundaries so CUDA's
  async kernel dispatch doesn't understate stage 3/4 on the `cuda` run.

## Results (ms), full precision in `results/latency_breakdown.csv`

| device | stage | mean | p50 | p95 | p99 | min | max |
|---|---|---|---|---|---|---|---|
| CPU | preprocessing | 0.255 | 0.183 | 0.630 | 0.864 | 0.061 | 1.153 |
| CPU | BAG construction | 0.029 | 0.022 | 0.059 | 0.082 | 0.007 | 0.124 |
| CPU | GATv2 forward | 1.511 | 1.449 | 1.917 | 2.297 | 1.117 | 2.663 |
| CPU | **end-to-end** | **2.014** | **1.845** | **2.891** | **3.474** | 1.329 | 4.413 |
| CUDA | preprocessing | 0.198 | 0.144 | 0.495 | 0.535 | 0.054 | 0.950 |
| CUDA | BAG construction | 0.022 | 0.017 | 0.046 | 0.055 | 0.005 | 0.096 |
| CUDA | GATv2 forward | 1.812 | 1.763 | 2.152 | 2.728 | 1.648 | 3.314 |
| CUDA | **end-to-end** | **2.310** | **2.183** | **2.835** | **3.576** | 1.966 | 4.776 |

Model load time (one-time, not per-request): CPU 59.3 ms, CUDA 151.8 ms
(CUDA context initialization overhead).

## Findings

- **GATv2 forward pass dominates end-to-end latency** on both devices
  (~75% of the total) — preprocessing and BAG construction together are
  under 15% of the budget, consistent with the graphs being small (mean
  21.6 nodes, 80.6 edges — see `docs/EXPERIMENT_LOG_semantic_edge_investigation.md`).
- **GPU is *slower* than CPU for single-request (batch_size=1) inference**
  (mean 2.31ms vs 2.01ms, p99 3.58ms vs 3.47ms) — for a graph this small,
  host↔device transfer and CUDA kernel-launch overhead outweigh the
  compute saved by the GPU. This is expected for single-sample GNN
  inference at this scale and is *not* a regression; it means a CPU-only
  WAF deployment is the better default for this model size, with GPU only
  paying off under high-throughput batched inference (not measured here).

## Comparison against the paper's existing "~3.7ms" claim

The paper (Abstract, §"Analysis of detection latency and operational
overhead", Table 2) currently states "*an average inference latency of
~3.7 ms per request*" / "*processes a single request in an average of
3.71 ms*", cited as supporting a real-time-deployment claim, without
specifying device, batch size, percentile, or hardware.

**The claim holds, and the measurement here is more favorable than 3.7ms on
mean, but the margin narrows sharply at the tail:**

| | CPU | CUDA |
|---|---|---|
| mean end-to-end | 2.01 ms (**46% faster** than 3.7ms) | 2.31 ms (**38% faster**) |
| p99 end-to-end | 3.47 ms (**still under** 3.7ms, 6% margin) | 3.58 ms (**still under**, 3% margin) |
| max observed (1000 requests) | 4.41 ms (**exceeds** 3.7ms) | 4.78 ms (**exceeds**) |

**Recommendation: keep the real-time claim, but rescope it with the
specifics this measurement provides** rather than the single unqualified
number:
- State the number as a **mean with a percentile**, not mean alone — "3.7ms"
  read as a p99 (not mean) is actually a *fair, slightly conservative*
  restatement of what's measured here (3.47-3.58ms p99), so the original
  number survives essentially unchanged if reframed as "p99 latency", not
  "average latency" as currently written.
- If kept as a *mean*, the true measured mean (2.0-2.3ms) is meaningfully
  better than 3.7ms — worth updating to the stronger, now-verified number
  rather than the old unverified one.
- Specify device (CPU **or** GPU — not both average to one number, since
  they differ and GPU is the slower one here), hardware model, batch_size=1,
  and N — all now available in this file and `results/latency_breakdown.csv`.
- Do not claim the number holds under batched/high-throughput serving
  without a separate batched-inference measurement — not done here.
