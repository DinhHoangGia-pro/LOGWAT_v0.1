"""Held-out E_sem window-boundary test: 2 cells specifically for the
context-distance (benign attribute padding) XSS mechanism -- see
`docs/EXPERIMENT_LOG_semantic_edge_investigation.md`, "XSS
Noise-Augmentation Feasibility Check" and "E_sem hard distance threshold".

Deliberately SEPARATE from `scripts/build_heldout_matrix_eval.py`'s 9-cell
matrix, for two reasons:
1. That script's `make_matrix_rows()` enforces a "no literal semantic-pair
   keyword" rule (`forbidden` list includes 'onload=', 'onerror=', etc.) to
   keep those 9 cells testing generalization AWAY from E_sem's keyword
   vocabulary. These 2 cells test the OPPOSITE thing on purpose: whether
   E_sem's window-based connectivity survives when the literal keywords ARE
   present but spaced apart -- they must contain 'svg'/'onload' literally to
   be a meaningful test at all. Reusing that script's validation would
   reject these rows outright, and weakening that check would compromise the
   original 9 cells' whole point.
2. Keeps the well-referenced, 9-row `results/heldout_matrix_full.csv` shape
   completely untouched -- other scripts/docs depend on it staying exactly
   9 rows.

Two cells, both using the ('svg','onload') pair with 10 distinct held-out
tag names never used in scripts/add_xss_context_augmentation.py's training
rows (article/aside/details/figcaption/marquee/summary/template/tfoot/bdi/
ruby -- checked disjoint from the training augmentation's sampled tags):

- `attribute_spacing_in_range`: 2 inserted benign attributes, token gap=12
  (<window=15) -- the case the new training augmentation targets. Expected
  to improve after training on scripts/add_xss_context_augmentation.py's
  rows.
- `attribute_spacing_out_of_range`: 6 inserted benign attributes, token
  gap=34 (>=16, past window) -- the proven-unfixable-by-data architectural
  limit. Expected to fail both BEFORE and AFTER the augmentation/retrain,
  as a control confirming the limit is real and training-independent.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader

from src.bag.graph_builder import build_single_graph
from src.models.logwat import HeavyWebGNN
from src.preprocessing.tokenizer import web_security_tokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
MODEL_PATH = DATA_DIR / 'models_pretrained' / 'best_web_gnn_seed42.pth'
RESULT_PATH = RESULTS_DIR / 'heldout_window_test.csv'
HELDOUT_CSV = DATA_DIR / 'heldout_window_test.csv'

TAGS = ['article', 'aside', 'details', 'figcaption', 'marquee', 'summary',
        'template', 'tfoot', 'bdi', 'ruby']


def make_rows():
    rows = []
    for i, tag in enumerate(TAGS):
        content = f'<svg><{tag} data-a{i}=v{i} id=q{i} onload=alert({i})></{tag}>'
        rows.append({'source_uid': f'window_test_in_range_{i:02d}', 'content': content,
                     'attack_type': 2, 'technique': 'attribute_spacing_in_range'})
    for i, tag in enumerate(TAGS):
        attrs = ' '.join(f'data-x{j}{i}=v{j}{i}' for j in range(6))
        content = f'<svg><{tag} {attrs} onload=alert({i})></{tag}>'
        rows.append({'source_uid': f'window_test_out_of_range_{i:02d}', 'content': content,
                     'attack_type': 2, 'technique': 'attribute_spacing_out_of_range'})
    df = pd.DataFrame(rows)

    # verify gap on every row before evaluating anything, per the same
    # per-row-checked discipline as add_xss_context_augmentation.py
    for _, row in df.iterrows():
        tokens = web_security_tokenizer(row['content'])
        gap = tokens.index('onload') - tokens.index('svg')
        expected_in_range = row['technique'] == 'attribute_spacing_in_range'
        if expected_in_range:
            assert gap < 15, f"{row['source_uid']}: expected gap<15, got {gap}"
        else:
            assert gap >= 16, f"{row['source_uid']}: expected gap>=16, got {gap}"
    return df


def evaluate(df, model_path):
    model = HeavyWebGNN(use_edge_attr=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    rows = []
    for technique in ['attribute_spacing_in_range', 'attribute_spacing_out_of_range']:
        subset = df[df['technique'] == technique]
        y_true = subset['attack_type'].astype(int).tolist()
        graphs = [build_single_graph(str(r['content']), source_uid=str(r['source_uid']),
                                      attack_type=int(r['attack_type']),
                                      use_seq=True, use_skip=True, use_sem=True, use_edge_attr=False)
                  for _, r in subset.iterrows()]
        y_pred = []
        with torch.no_grad():
            for batch in DataLoader(graphs, batch_size=8, shuffle=False):
                logits = model(batch.x, batch.edge_index, batch.batch, edge_attr=None)
                y_pred.extend(int(p) for p in logits.argmax(dim=1).tolist())
        rows.append({
            'label': 2, 'class_name': 'XSS', 'technique': technique, 'n_samples': len(subset),
            'gatv2_accuracy': float(accuracy_score(y_true, y_pred)),
            'gatv2_correct': int(sum(1 for a, b in zip(y_true, y_pred) if a == b)),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=str(MODEL_PATH))
    parser.add_argument('--result_path', default=str(RESULT_PATH))
    parser.add_argument('--label', default='', help='free-text tag appended to printed output, e.g. PRE/POST')
    args = parser.parse_args()

    df = make_rows()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(HELDOUT_CSV, index=False)

    results = evaluate(df, args.model_path)
    Path(args.result_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.result_path, index=False)
    tag = f' [{args.label}]' if args.label else ''
    print(f"=== E_sem window-boundary held-out test{tag} (model={args.model_path}) ===")
    print(results.to_string(index=False))


if __name__ == '__main__':
    main()
