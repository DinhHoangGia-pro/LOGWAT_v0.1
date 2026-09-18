"""Per-request latency breakdown, batch_size=1 (single-request WAF
simulation, not training-style batching), on CPU and GPU (if available).

Uses the exact production code path (`web_security_tokenizer`,
`get_node_features`, `sequential_edges`, `skip_edges`, `semantic_edges`,
`HeavyWebGNN`) decomposed into 3 timed stages, not a reimplementation:
  1. Preprocessing: tokenize + per-token node features
  2. BAG construction: E_seq + E_skip + E_sem edge construction
  3. GATv2 forward pass (includes host->device transfer for the GPU run,
     since that is a real per-request cost in a WAF deployment)
  4. End-to-end total (sum of 1-3, measured as one continuous wall-clock
     span per request, not just the sum of the 3 medians)

N=1000 real request contents, sampled (with replacement) from the frozen
test split, so the benchmark reflects realistic content-length variance
rather than one fixed string. 20 warm-up iterations (untimed) precede the
1000 measured ones on each device, standard practice to exclude first-call/
CUDA-context-init overhead. Model load time measured once, separately, not
counted per-request.

Writes results/latency_breakdown.csv (mean/p50/p95/p99/min/max per stage
per device) and prints the full environment (torch/torch_geometric/CUDA
versions, CPU model, GPU name) needed to interpret the numbers.
"""
import pickle
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Data

from src.bag.node_features import get_node_features
from src.edges.sequential import sequential_edges
from src.edges.semantic import semantic_edges
from src.edges.skip import skip_edges
from src.models.logwat import HeavyWebGNN
from src.preprocessing.tokenizer import web_security_tokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
MODEL_PATH = DATA_DIR / 'models_pretrained' / 'best_web_gnn_seed42.pth'
AUGMENTED_CSV = DATA_DIR / 'augmented_web_attack.csv'
TEST_SPLIT_PKL = DATA_DIR / 'test_split_indices.pkl'
OUT_CSV = RESULTS_DIR / 'latency_breakdown.csv'

N_REQUESTS = 1000
N_WARMUP = 20
SEED = 42


def get_cpu_model():
    try:
        out = subprocess.run(['grep', 'model name', '/proc/cpuinfo'], capture_output=True, text=True)
        line = out.stdout.strip().split('\n')[0]
        return line.split(':', 1)[1].strip()
    except Exception:
        return platform.processor() or 'unknown'


def build_content_pool():
    import pandas as pd
    df = pd.read_csv(AUGMENTED_CSV)
    with open(TEST_SPLIT_PKL, 'rb') as f:
        split = pickle.load(f)
    test_idx = split['test_idx']
    contents = df.iloc[test_idx]['content'].astype(str).tolist()
    rng = np.random.RandomState(SEED)
    sampled = rng.choice(contents, size=N_REQUESTS, replace=True).tolist()
    return sampled


def time_one_request(content, model, device):
    t0 = time.perf_counter()

    # --- Stage 1: preprocessing ---
    tokens = web_security_tokenizer(content)
    num_nodes = len(tokens)
    x_list = [get_node_features(t, i, num_nodes) for i, t in enumerate(tokens)]
    t1 = time.perf_counter()

    # --- Stage 2: BAG construction ---
    seq = sequential_edges(tokens)
    skip = skip_edges(tokens, k=2)
    sem = semantic_edges(tokens, window=15)
    edges = seq + skip + sem
    if not edges:
        edges = [(0, 0)]
    t2 = time.perf_counter()

    # --- assemble Data object (not separately timed -- cheap tensor-construction
    #     overhead shared by both stage 2's output and stage 3's input) ---
    x = torch.tensor(x_list, dtype=torch.float) if x_list else torch.zeros((1, 64), dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index, y=torch.tensor([0], dtype=torch.long))
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

    # --- Stage 3: GATv2 forward pass (incl. host->device transfer) ---
    data = data.to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t3 = time.perf_counter()
    with torch.no_grad():
        _ = model(data.x, data.edge_index, data.batch, edge_attr=None)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t4 = time.perf_counter()

    return {
        'preprocessing_ms': (t1 - t0) * 1000,
        'bag_construction_ms': (t2 - t1) * 1000,
        'gatv2_forward_ms': (t4 - t3) * 1000,
        'end_to_end_ms': (t4 - t0) * 1000,
    }


def run_device(device_name, contents):
    device = torch.device(device_name)
    t_load0 = time.perf_counter()
    model = HeavyWebGNN(use_edge_attr=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    load_time_ms = (time.perf_counter() - t_load0) * 1000

    for c in contents[:N_WARMUP]:
        time_one_request(c, model, device)

    rows = []
    for c in contents:
        rows.append(time_one_request(c, model, device))

    df = pd.DataFrame(rows)
    summary = []
    for col in df.columns:
        vals = df[col].values
        summary.append({
            'device': device_name, 'stage': col,
            'mean_ms': vals.mean(), 'p50_ms': np.percentile(vals, 50),
            'p95_ms': np.percentile(vals, 95), 'p99_ms': np.percentile(vals, 99),
            'min_ms': vals.min(), 'max_ms': vals.max(),
        })
    return summary, load_time_ms


def main():
    print("=== Environment ===")
    print(f"torch: {torch.__version__}")
    print(f"torch_geometric: {torch_geometric.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version (torch.version.cuda): {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CPU model: {get_cpu_model()}")
    print(f"N_REQUESTS={N_REQUESTS}, N_WARMUP={N_WARMUP}, content sampled from frozen test split (seed={SEED})")
    print()

    contents = build_content_pool()

    all_summaries = []
    load_times = {}

    print(f"--- Running on CPU ({N_REQUESTS} requests + {N_WARMUP} warmup) ---")
    summary_cpu, load_cpu = run_device('cpu', contents)
    all_summaries.extend(summary_cpu)
    load_times['cpu'] = load_cpu
    print(f"CPU model load time: {load_cpu:.2f} ms")

    if torch.cuda.is_available():
        print(f"--- Running on CUDA ({N_REQUESTS} requests + {N_WARMUP} warmup) ---")
        summary_gpu, load_gpu = run_device('cuda', contents)
        all_summaries.extend(summary_gpu)
        load_times['cuda'] = load_gpu
        print(f"GPU model load time: {load_gpu:.2f} ms")
    else:
        print("CUDA not available -- skipping GPU benchmark.")

    result_df = pd.DataFrame(all_summaries)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_CSV, index=False)

    print("\n=== Latency breakdown (ms), model load time excluded ===")
    print(result_df.to_string(index=False))
    print(f"\nModel load times (separate, one-time cost, not per-request): {load_times}")
    print(f"\nWrote: {OUT_CSV}")


if __name__ == '__main__':
    main()
