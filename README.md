# LOGWAT

Web attack (SQLi/XSS) detection using a heterogeneous, semantic-aware graph
(BAG) + GATv2, for the Applied Intelligence paper submission.

## Pipeline

```
csic_database.csv (raw)
  -> scripts/prepare_data.py        (labeling + class balancing/augmentation)
  -> data/augmented_web_attack.csv
  -> scripts/build_graphs.py        (BAG construction: sequential/skip/semantic edges)
  -> data/web_graphs.pkl
  -> scripts/train_logwat.py        (GATv2, seed=42, fixed 3-class split)
  -> data/models_pretrained/best_web_gnn_seed42.pth
  -> scripts/evaluate.py
  -> results/main_results.csv
```

See [docs/DATASET.md](docs/DATASET.md) for the frozen dataset's schema/state and
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the exact commands, seed, and
checkpoint-by-checkpoint retrain history.

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install torch_scatter==2.1.2+pt26cu124 torch_sparse==0.6.18+pt26cu124 \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

## Running

The dataset (`data/augmented_web_attack.csv`) and the train/test split
(`data/test_split_indices.pkl`) are **frozen** (`dataset-v1-frozen` tag) — do not
regenerate them. To evaluate the already-trained checkpoint:

```bash
python scripts/evaluate.py
```

To retrain from the frozen, already-augmented dataset:

```bash
python scripts/build_graphs.py
python scripts/train_logwat.py
python scripts/evaluate.py
```

Full from-scratch sequence (including the append-only augmentation steps already
baked into the frozen dataset) and the ablation-study commands are in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Investigation log

[docs/EXPERIMENT_LOG_semantic_edge_investigation.md](docs/EXPERIMENT_LOG_semantic_edge_investigation.md)
records the full investigation into a benchmark/generalization gap discovered during
this work: near-perfect accuracy on the frozen test split vs. failures on a held-out
matrix of unseen evasion techniques, the root-cause diagnosis, the fixes attempted,
and a 6-configuration ablation isolating which BAG edge types are load-bearing for
each evaluation regime.

## Other docs

- [docs/DATASET.md](docs/DATASET.md) — dataset schema, frozen state, known data gaps.
- [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) — threat model.
- [docs/LATENCY.md](docs/LATENCY.md) — latency notes.

## License

[MIT](LICENSE)
