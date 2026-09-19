"""Per-request latency breakdown for the Group-A graph baselines (#16), same
methodology as scripts/benchmark_latency.py (#11/#21): batch_size=1, CPU and GPU,
N=1000 requests sampled (seed 42, with replacement) from the frozen test split,
20 untimed warm-up requests, model load time measured separately, 3 timed stages
+ end-to-end:

  1. preprocessing:    tokenize + per-token node features
  2. bag_construction: E_seq + E_skip + E_sem edges (+ the relation one-hot
                       edge_attr, for HGT only -- it needs it, so it is HGT's cost)
  3. forward:          host->device transfer + model forward
  4. end_to_end:       one continuous wall-clock span per request

The deployed GATv2 is measured IN THE SAME RUN as a reference row, so every model
sees the same machine state (latency depends on background load, so numbers from
different sessions -- e.g. results/latency_breakdown.csv -- are not directly
comparable). Run it with the GPU otherwise idle (not during a training run). Deliberately does
NOT call set_seed(): that enables torch.use_deterministic_algorithms(True), whose
slower scatter kernels would inflate GPU forward times relative to #11/#21.

Writes results/latency_breakdown_graph_baselines.csv
(model, device, stage, mean_ms, p50_ms, p95_ms, p99_ms, min_ms, max_ms).
"""
import time

import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Data

from scripts.benchmark_latency import (N_REQUESTS, N_WARMUP, RESULTS_DIR, build_content_pool, get_cpu_model)
from scripts.evaluate_baselines import load_model
from src.bag.graph_builder import RELATION_TYPES
from src.bag.node_features import get_node_features
from src.edges.semantic import semantic_edges
from src.edges.sequential import sequential_edges
from src.edges.skip import skip_edges
from src.models.baselines_graph import GRAPH_BASELINES
from src.models.logwat import HeavyWebGNN
from src.preprocessing.tokenizer import web_security_tokenizer
from scripts.benchmark_latency import MODEL_PATH as GATV2_PATH

OUT_CSV = RESULTS_DIR / 'latency_breakdown_graph_baselines.csv'
SEED = 42
STAGES = ['preprocessing_ms', 'bag_construction_ms', 'forward_ms', 'end_to_end_ms']


def time_one_request(content, model, device, use_edge_attr):
    t0 = time.perf_counter()
    tokens = web_security_tokenizer(content)
    num_nodes = len(tokens)
    x_list = [get_node_features(t, i, num_nodes) for i, t in enumerate(tokens)]
    t1 = time.perf_counter()

    seq = sequential_edges(tokens)
    skip = skip_edges(tokens, k=2)
    sem = semantic_edges(tokens, window=15)
    edges = seq + skip + sem
    relations = ['seq'] * len(seq) + ['skip'] * len(skip) + ['sem'] * len(sem)
    if not edges:
        edges, relations = [(0, 0)], ['seq']
    edge_attr = None
    if use_edge_attr:
        rel_idx = {r: i for i, r in enumerate(RELATION_TYPES)}
        edge_attr = torch.zeros((len(relations), len(RELATION_TYPES)), dtype=torch.float)
        for e, r in enumerate(relations):
            edge_attr[e, rel_idx[r]] = 1.0
    t2 = time.perf_counter()

    x = torch.tensor(x_list, dtype=torch.float) if x_list else torch.zeros((1, 64), dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index, y=torch.tensor([0], dtype=torch.long))
    if edge_attr is not None:
        data.edge_attr = edge_attr
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

    data = data.to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t3 = time.perf_counter()
    with torch.no_grad():
        model(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr if use_edge_attr else None)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t4 = time.perf_counter()
    return {'preprocessing_ms': (t1 - t0) * 1000, 'bag_construction_ms': (t2 - t1) * 1000,
            'forward_ms': (t4 - t3) * 1000, 'end_to_end_ms': (t4 - t0) * 1000}


def load_gatv2(device):
    m = HeavyWebGNN(use_edge_attr=False)
    m.load_state_dict(torch.load(GATV2_PATH, map_location=device))
    return m.to(device).eval()


def run(name, device_name, contents):
    device = torch.device(device_name)
    t_load0 = time.perf_counter()
    if name == 'GATv2':
        model = load_gatv2(device)
        use_ea = False
    else:
        key = {'GCN': 'gcn', 'GraphSAGE': 'graphsage', 'GIN': 'gin', 'HGT': 'hgt'}[name]
        model = load_model(key, SEED, device)
        use_ea = GRAPH_BASELINES[key].needs_edge_attr
    if device.type == 'cuda':
        torch.cuda.synchronize()
    load_ms = (time.perf_counter() - t_load0) * 1000
    for c in contents[:N_WARMUP]:
        time_one_request(c, model, device, use_ea)
    df = pd.DataFrame([time_one_request(c, model, device, use_ea) for c in contents])
    rows = [{'model': name, 'device': device_name, 'stage': col, 'mean_ms': df[col].mean(),
             'p50_ms': np.percentile(df[col], 50), 'p95_ms': np.percentile(df[col], 95),
             'p99_ms': np.percentile(df[col], 99), 'min_ms': df[col].min(), 'max_ms': df[col].max()}
            for col in STAGES]
    return rows, load_ms


def main():
    print("=== Environment ===")
    print(f"torch: {torch.__version__} | torch_geometric: {torch_geometric.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}")
    print(f"CPU model: {get_cpu_model()}")
    print(f"N_REQUESTS={N_REQUESTS}, N_WARMUP={N_WARMUP}, content sampled from frozen test split (seed={SEED})\n")

    contents = build_content_pool()
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    all_rows, loads = [], {}
    for device_name in devices:
        for name in ['GATv2', 'GCN', 'GraphSAGE', 'GIN', 'HGT']:
            rows, load_ms = run(name, device_name, contents)
            all_rows += rows
            loads[(name, device_name)] = load_ms
            e2e = [r for r in rows if r['stage'] == 'end_to_end_ms'][0]
            print(f"[{device_name}] {name:10s} end_to_end mean={e2e['mean_ms']:.3f} ms p50={e2e['p50_ms']:.3f} p95={e2e['p95_ms']:.3f} | load={load_ms:.1f} ms")

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT_CSV, index=False)
    print("\n=== Latency breakdown (ms), mean per stage ===")
    print(out.pivot_table(index=['device', 'model'], columns='stage', values='mean_ms', sort=False)[STAGES].round(3).to_string())
    print(f"\nModel load times (ms, one-time, excluded above): { {f'{k[0]}/{k[1]}': round(v, 1) for k, v in loads.items()} }")
    print(f"Wrote: {OUT_CSV}")


if __name__ == '__main__':
    main()
