"""Evaluate the #16 baselines on the same 3 tests already run for GATv2 /
RoBERTa / CodeBERT:

  a) frozen test split (test_split_indices.pkl['test_idx']) -- plus the
     duplicate/novel-content accuracy breakdown (25.2% of the test split shares
     byte-identical content with train; see EXPERIMENT_LOG "Test-split content
     duplication")
  b) held-out 9-cell matrix (scripts/build_heldout_matrix_eval.make_matrix_rows(),
     the exact same 90 rows)
  c) external dataset (data/external/external_dataset_clean.csv)

Two groups, two output files (different input kinds), same long-format schema as
results/transformer_baselines.csv so they merge into final_baseline_comparison.csv:

  --group graph     GCN / GraphSAGE / GIN / HGT      -> results/graph_baselines.csv
  --group sequence  Bi-LSTM / TextCNN / StackLSTM    -> results/sequence_baselines.csv

  method, eval_mode, group, precision, recall, f1_score, support,
  accuracy, n_correct, n_total

The canonical CSVs hold seed 42 (the seed of every existing GATv2 / transformer
number). `--seeds 42 43 44 45 46` additionally writes
results/<group>_baselines_5seed.csv: one row per (method, seed) with the headline
metrics, for a mean+-std against #22's GATv2 5-seed statistics. Checkpoints are
data/models_pretrained/baseline_<model>_seed<seed>.pth (scripts/train_baseline.py).

Backs up any existing output CSV to <name>_PRE_<timestamp>.csv before writing.
Rows of methods NOT being re-evaluated in this call are kept as they were, so
baselines can be evaluated and committed one at a time.
"""
import argparse
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch_geometric.loader import DataLoader

from scripts.build_heldout_matrix_eval import LABELS, TECHNIQUES, make_matrix_rows
from scripts.train_baseline import baseline_spec, ckpt_path
from src.baselines.sequence_data import build_sequence_data, load_vocab
from src.bag.graph_builder import build_single_graph
from src.models.baselines_graph import GRAPH_BASELINES
from src.models.baselines_seq import SEQ_BASELINES

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
AUGMENTED_CSV = DATA_DIR / 'augmented_web_attack.csv'
SPLIT_PKL = DATA_DIR / 'test_split_indices.pkl'
EXTERNAL_CSV = DATA_DIR / 'external' / 'external_dataset_clean.csv'
EXTERNAL_GRAPHS = DATA_DIR / 'external' / 'external_dataset_graphs.pkl'
EXTERNAL_GRAPHS_EA = DATA_DIR / 'external' / 'external_dataset_graphs_edge_attr.pkl'

CLASS_NAMES = ['Benign', 'SQLi', 'XSS']
COLUMNS = ['method', 'eval_mode', 'group', 'precision', 'recall', 'f1_score',
           'support', 'accuracy', 'n_correct', 'n_total']

GROUPS = {
    'graph': {'out': RESULTS_DIR / 'graph_baselines.csv',
              'out_seeds': RESULTS_DIR / 'graph_baselines_5seed.csv',
              'methods': {'GCN': 'gcn', 'GraphSAGE': 'graphsage', 'GIN': 'gin', 'HGT': 'hgt'}},
    'sequence': {'out': RESULTS_DIR / 'sequence_baselines.csv',
                 'out_seeds': RESULTS_DIR / 'sequence_baselines_5seed.csv',
                 'methods': {'Bi-LSTM': 'bilstm', 'TextCNN': 'textcnn', 'StackLSTM': 'stacklstm'}},
}

_pkl_cache = {}


def load_graphs(path):
    path = str(path)
    if path not in _pkl_cache:
        with open(path, 'rb') as f:
            _pkl_cache[path] = pickle.load(f)['graphs']
    return _pkl_cache[path]


def load_model(model_key, seed, device):
    factory, _ = baseline_spec(model_key)
    model = factory(False)
    model.load_state_dict(torch.load(ckpt_path(model_key, seed), map_location=device))
    return model.to(device).eval()


def needs_edge_attr(model_key):
    return model_key in GRAPH_BASELINES and GRAPH_BASELINES[model_key].needs_edge_attr


@torch.no_grad()
def predict(model, data_list, use_edge_attr, device, batch_size=256):
    preds = []
    for batch in DataLoader(data_list, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        ea = batch.edge_attr if use_edge_attr else None
        preds.extend(model(batch.x, batch.edge_index, batch.batch, edge_attr=ea).argmax(dim=1).cpu().tolist())
    return preds


def report_rows(method, eval_mode, y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    rows = []
    for key, m in report.items():
        if not isinstance(m, dict):
            continue
        rows.append({'method': method, 'eval_mode': eval_mode, 'group': key,
                     'precision': m['precision'], 'recall': m['recall'], 'f1_score': m['f1-score'],
                     'support': m['support'], 'accuracy': None, 'n_correct': None, 'n_total': None})
    return rows, report


def acc_row(method, eval_mode, group, y_true, y_pred):
    n_correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return {'method': method, 'eval_mode': eval_mode, 'group': group, 'precision': None, 'recall': None,
            'f1_score': None, 'support': None, 'accuracy': n_correct / len(y_true),
            'n_correct': n_correct, 'n_total': len(y_true)}


def texts_to_data(model_key, texts, uids, labels):
    if model_key in SEQ_BASELINES:
        vocab = load_vocab()
        return [build_sequence_data(t, vocab, source_uid=u, attack_type=y) for t, u, y in zip(texts, uids, labels)]
    ea = needs_edge_attr(model_key)
    return [build_single_graph(t, source_uid=u, attack_type=y, use_edge_attr=ea)
            for t, u, y in zip(texts, uids, labels)]


def evaluate_method(method, model_key, seed, device):
    """-> (rows in the transformer_baselines schema, headline-metrics dict)"""
    model = load_model(model_key, seed, device)
    use_ea = needs_edge_attr(model_key)
    rows, summary = [], {'method': method, 'seed': seed}

    # --- a) frozen test split ---
    _, data_path = baseline_spec(model_key)
    graphs = load_graphs(data_path)
    with open(SPLIT_PKL, 'rb') as f:
        test_idx = list(pickle.load(f)['test_idx'])
    test_set = set(test_idx)
    train_idx = [i for i in range(len(graphs)) if i not in test_set]
    y_true = [int(graphs[i].y.item()) for i in test_idx]
    y_pred = predict(model, [graphs[i] for i in test_idx], use_ea, device)
    r, rep = report_rows(method, 'test_split', y_true, y_pred)
    rows += r
    summary.update(test_acc=rep['accuracy'], test_macro_f1=rep['macro avg']['f1-score'])

    contents = pd.read_csv(AUGMENTED_CSV)['content'].astype(str).tolist()
    train_contents = {contents[i] for i in train_idx}
    dup = [contents[i] in train_contents for i in test_idx]
    for flag, name in [(True, 'duplicate_content'), (False, 'novel_content')]:
        sel = [k for k, d in enumerate(dup) if d == flag]
        if sel:
            rows.append(acc_row(method, 'test_split_duplication_breakdown', name,
                                [y_true[k] for k in sel], [y_pred[k] for k in sel]))

    # --- b) held-out 9-cell matrix ---
    df = make_matrix_rows()
    n_correct_total = n_total = 0
    for label in [0, 1, 2]:
        for technique in TECHNIQUES[label]:
            sub = df[(df['class_name'] == LABELS[label]) & (df['technique'] == technique)]
            if sub.empty:
                continue
            data = texts_to_data(model_key, sub['content'].astype(str).tolist(),
                                 sub['source_uid'].astype(str).tolist(), sub['attack_type'].astype(int).tolist())
            yt = sub['attack_type'].astype(int).tolist()
            row = acc_row(method, 'held_out_matrix', f"{LABELS[label]}/{technique}",
                          yt, predict(model, data, use_ea, device, batch_size=8))
            rows.append(row)
            n_correct_total += row['n_correct']
            n_total += row['n_total']
    summary.update(heldout_correct=n_correct_total, heldout_total=n_total,
                   heldout_full_cells=sum(1 for r_ in rows if r_['eval_mode'] == 'held_out_matrix' and r_['accuracy'] == 1.0))

    # --- c) external dataset ---
    ext = pd.read_csv(EXTERNAL_CSV)
    y_ext = ext['attack_type'].astype(int).tolist()
    if model_key in SEQ_BASELINES:
        ext_data = texts_to_data(model_key, ext['content'].astype(str).tolist(),
                                 [None] * len(ext), y_ext)
    else:
        ext_data = load_graphs(EXTERNAL_GRAPHS_EA if use_ea else EXTERNAL_GRAPHS)
        assert len(ext_data) == len(ext), f"external graphs/csv mismatch: {len(ext_data)} vs {len(ext)}"
        assert [int(g.y.item()) for g in ext_data] == y_ext, "external graph labels differ from csv labels"
    y_ext_pred = predict(model, ext_data, use_ea, device)
    r, rep = report_rows(method, 'external_dataset', y_ext, y_ext_pred)
    rows += r
    summary.update(ext_acc=rep['accuracy'], ext_macro_f1=rep['macro avg']['f1-score'],
                   ext_weighted_f1=rep['weighted avg']['f1-score'],
                   ext_benign_recall=rep['Benign']['recall'], ext_sqli_recall=rep['SQLi']['recall'],
                   ext_xss_recall=rep['XSS']['recall'])
    return rows, summary


def write_merged(out_csv, new_rows, method_order):
    """Replace rows of the re-evaluated methods, keep the other methods' rows."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(new_rows, columns=COLUMNS)
    if out_csv.exists():
        backup = out_csv.with_name(f"{out_csv.stem}_PRE_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        shutil.copy(out_csv, backup)
        print(f"[*] backed up existing {out_csv.name} -> {backup.name}")
        old = pd.read_csv(out_csv)
        new_df = pd.concat([old[~old['method'].isin(new_df['method'].unique())], new_df], ignore_index=True)
    new_df['_o'] = new_df['method'].map({m: i for i, m in enumerate(method_order)})
    new_df = new_df.sort_values('_o', kind='stable').drop(columns='_o')
    new_df.to_csv(out_csv, index=False)
    print(f"[+] wrote {len(new_df)} rows to {out_csv}")
    return new_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True, choices=list(GROUPS))
    parser.add_argument('--methods', nargs='*', help="subset of method names (default: whole group)")
    parser.add_argument('--seeds', nargs='*', type=int, default=[42])
    args = parser.parse_args()

    cfg = GROUPS[args.group]
    methods = args.methods or list(cfg['methods'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    canonical_rows, per_seed = [], []
    for method in methods:
        key = cfg['methods'][method]
        for seed in args.seeds:
            if not ckpt_path(key, seed).exists():
                print(f"[!] {method} seed {seed}: no checkpoint at {ckpt_path(key, seed)}, skipping")
                continue
            rows, summary = evaluate_method(method, key, seed, device)
            per_seed.append(summary)
            print(f"[{method}][seed {seed}] " + ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items() if k not in ('method', 'seed')))
            if seed == 42:
                canonical_rows += rows

    if canonical_rows:
        df = write_merged(cfg['out'], canonical_rows, list(cfg['methods']))
        print(df[df['eval_mode'].isin(['held_out_matrix'])].pivot_table(
            index='group', columns='method', values='n_correct', sort=False).to_string())
    if len(args.seeds) > 1 and per_seed:
        ps = pd.DataFrame(per_seed)
        ps.to_csv(cfg['out_seeds'], index=False)
        print(f"[+] wrote per-seed summary to {cfg['out_seeds']}")
        print(ps.groupby('method', sort=False).agg(['mean', 'std']).drop(columns='seed').T.to_string())


if __name__ == '__main__':
    main()
