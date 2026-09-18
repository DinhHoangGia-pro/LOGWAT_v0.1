"""Per-request latency breakdown for the RoBERTa/CodeBERT baselines,
same methodology as scripts/benchmark_latency.py (#11, GATv2): batch_size=1,
N=1000 requests, 20 untimed warm-up iterations, CPU and CUDA measured
separately, mean/p50/p95/p99/min/max per stage. Reuses that script's exact
content pool (`build_content_pool()`, same N/seed/source) so the two
benchmarks are sampling the identical 1000 requests.

Stages timed (no BAG-construction stage -- these are plain text
classifiers, not graph models):
  1. Tokenize: raw content -> input_ids/attention_mask
  2. Forward pass (incl. host->device transfer for the GPU run, a real
     per-request cost in a WAF deployment)
  3. End-to-end total (one continuous wall-clock span per request)

Appends to results/latency_breakdown.csv (method column added; existing
GATv2 rows backfilled with method=GATv2, not otherwise altered) rather than
overwriting it. Backs up the pre-existing file first.
"""
import argparse
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers

from scripts.benchmark_latency import build_content_pool, get_cpu_model, N_REQUESTS, N_WARMUP
from src.baselines.transformer_common import load_finetuned

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
OUT_CSV = RESULTS_DIR / 'latency_breakdown.csv'

METHODS = {
    'RoBERTa': DATA_DIR / 'models_pretrained' / 'roberta_baseline_seed42',
    'CodeBERT': DATA_DIR / 'models_pretrained' / 'codebert_baseline_seed42',
}


def time_one_request(content, model, tokenizer, max_length, device):
    t0 = time.perf_counter()

    enc = tokenizer(content, padding='max_length', truncation=True,
                     max_length=max_length, return_tensors='pt')
    t1 = time.perf_counter()

    enc = {k: v.to(device) for k, v in enc.items()}
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    with torch.no_grad():
        _ = model(**enc)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t3 = time.perf_counter()

    return {
        'tokenize_ms': (t1 - t0) * 1000,
        'forward_ms': (t3 - t2) * 1000,
        'end_to_end_ms': (t3 - t0) * 1000,
    }


def run_device(method, output_dir, device_name, contents):
    device = torch.device(device_name)
    t_load0 = time.perf_counter()
    model, tokenizer, max_length, _ = load_finetuned(output_dir, device=device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    load_time_ms = (time.perf_counter() - t_load0) * 1000

    for c in contents[:N_WARMUP]:
        time_one_request(c, model, tokenizer, max_length, device)

    rows = []
    for c in contents:
        rows.append(time_one_request(c, model, tokenizer, max_length, device))

    df = pd.DataFrame(rows)
    summary = []
    for col in df.columns:
        vals = df[col].values
        summary.append({
            'method': method, 'device': device_name, 'stage': col,
            'mean_ms': vals.mean(), 'p50_ms': np.percentile(vals, 50),
            'p95_ms': np.percentile(vals, 95), 'p99_ms': np.percentile(vals, 99),
            'min_ms': vals.min(), 'max_ms': vals.max(),
        })
    return summary, load_time_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='*', default=list(METHODS.keys()))
    args = parser.parse_args()

    print("=== Environment ===")
    print(f"torch: {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version (torch.version.cuda): {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CPU model: {get_cpu_model()}")
    print(f"N_REQUESTS={N_REQUESTS}, N_WARMUP={N_WARMUP} (same content pool as scripts/benchmark_latency.py)")

    contents = build_content_pool()

    if OUT_CSV.exists():
        backup = RESULTS_DIR / f"latency_breakdown_PRE_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        shutil.copy(OUT_CSV, backup)
        print(f"[*] backed up existing {OUT_CSV} -> {backup}")
        existing = pd.read_csv(OUT_CSV)
        if 'method' not in existing.columns:
            existing.insert(0, 'method', 'GATv2')
            print("[*] backfilled method=GATv2 on pre-existing rows (schema now includes method)")
    else:
        existing = pd.DataFrame()

    all_summaries = []
    for method in args.methods:
        output_dir = METHODS[method]
        if not output_dir.exists():
            print(f"[!] {method} checkpoint not found at {output_dir}, skipping")
            continue

        print(f"\n--- {method}: CPU ({N_REQUESTS} requests + {N_WARMUP} warmup) ---")
        summary_cpu, load_cpu = run_device(method, output_dir, 'cpu', contents)
        all_summaries.extend(summary_cpu)
        print(f"{method} CPU model load time: {load_cpu:.2f} ms")

        if torch.cuda.is_available():
            print(f"--- {method}: CUDA ({N_REQUESTS} requests + {N_WARMUP} warmup) ---")
            summary_gpu, load_gpu = run_device(method, output_dir, 'cuda', contents)
            all_summaries.extend(summary_gpu)
            print(f"{method} GPU model load time: {load_gpu:.2f} ms")

    new_df = pd.DataFrame(all_summaries)
    result_df = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_CSV, index=False)

    print("\n=== Latency breakdown (ms), new rows only ===")
    print(new_df.to_string(index=False))
    print(f"\nWrote (appended): {OUT_CSV}")


if __name__ == '__main__':
    main()
