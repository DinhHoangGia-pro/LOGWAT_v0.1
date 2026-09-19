"""Evaluate the already-trained checkpoint on the external dataset. NO
training/fine-tuning happens here -- inference and two baselines only.

1. GATv2 (loaded checkpoint, use_edge_attr=False -- matches
   build_external_graphs.py and the checkpoint config confirmed in
   docs/REPRODUCIBILITY.md).
2. String-matching baseline: src.preprocessing.normalization.classify_request().
3. Decision-Tree size-only baseline ([num_nodes, num_edges]): trained on the
   SAME train_idx as scripts/decision_tree_size_baseline.py (from the current
   web_graphs.pkl / test_split_indices.pkl), predicting on the external
   graphs -- same method/seed as that script, applied to a different test set.

Writes results/external_dataset_evaluation.csv with one row per
(method, class-or-avg, metric).
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from torch_geometric.loader import DataLoader

from src.models.logwat import HeavyWebGNN
from src.preprocessing.normalization import classify_request

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
EXTERNAL_DIR = DATA_DIR / 'external'
RESULTS_DIR = ROOT / 'results'

EXTERNAL_CLEAN_CSV = EXTERNAL_DIR / 'external_dataset_clean.csv'
EXTERNAL_GRAPHS_PKL = EXTERNAL_DIR / 'external_dataset_graphs.pkl'
MODEL_PATH = DATA_DIR / 'models_pretrained' / 'best_web_gnn_seed42.pth'
TRAIN_GRAPHS_PKL = DATA_DIR / 'web_graphs.pkl'
TRAIN_SPLIT_PKL = DATA_DIR / 'test_split_indices.pkl'
OUT_CSV = RESULTS_DIR / 'external_dataset_evaluation.csv'
OUT_TXT = EXTERNAL_DIR / 'external_dataset_evaluation_report.txt'

CLASS_NAMES = ['Benign', 'SQLi', 'XSS']


def report_to_rows(method, y_true, y_pred, labels=(0, 1, 2)):
    report = classification_report(y_true, y_pred, labels=list(labels), target_names=CLASS_NAMES,
                                    output_dict=True, zero_division=0)
    rows = []
    for key, metrics in report.items():
        if not isinstance(metrics, dict):
            continue
        rows.append({
            'method': method,
            'label': key,
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'f1_score': metrics.get('f1-score'),
            'support': metrics.get('support'),
        })
    return rows


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(EXTERNAL_CLEAN_CSV)
    y_true_all = df['attack_type'].astype(int).tolist()

    with open(EXTERNAL_GRAPHS_PKL, 'rb') as f:
        ext_graphs = pickle.load(f)['graphs']
    assert len(ext_graphs) == len(df), f"graph/csv row mismatch: {len(ext_graphs)} vs {len(df)}"

    use_edge_attr = getattr(ext_graphs[0], 'edge_attr', None) is not None
    assert use_edge_attr is False, "external graphs must be built with use_edge_attr=False to match the checkpoint"

    # --- 1. GATv2 inference ---
    model = HeavyWebGNN(use_edge_attr=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    loader = DataLoader(ext_graphs, batch_size=256, shuffle=False)
    y_pred_gnn = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch, edge_attr=None)
            y_pred_gnn.extend(out.argmax(dim=1).cpu().numpy().tolist())

    cm_gnn = confusion_matrix(y_true_all, y_pred_gnn, labels=[0, 1, 2])
    rows = report_to_rows('gatv2', y_true_all, y_pred_gnn)

    # --- 2. String-matching baseline ---
    y_pred_str = [classify_request(c) for c in df['content'].tolist()]
    # classify_request() can return -1 (unknown) or 3 (other); map anything
    # outside {0,1,2} to a value guaranteed wrong for every true label so it
    # counts as an error in the report rather than crashing classification_report.
    y_pred_str_mapped = [p if p in (0, 1, 2) else -1 for p in y_pred_str]
    cm_str = confusion_matrix(y_true_all, y_pred_str_mapped, labels=[0, 1, 2])
    rows += report_to_rows('string_matching', y_true_all, y_pred_str_mapped)
    n_unknown_or_other = sum(1 for p in y_pred_str if p not in (0, 1, 2))

    # --- 3. Decision Tree size-only baseline (same method/seed as
    #     scripts/decision_tree_size_baseline.py, trained on the SAME
    #     train_idx, predicting on the external graphs instead of test_idx) ---
    with open(TRAIN_GRAPHS_PKL, 'rb') as f:
        train_graphs = pickle.load(f)['graphs']
    with open(TRAIN_SPLIT_PKL, 'rb') as f:
        split = pickle.load(f)
    train_idx = [i for i in range(len(train_graphs)) if i not in set(split['test_idx'])]

    def features_labels(graphs, idx_list=None):
        gs = graphs if idx_list is None else [graphs[i] for i in idx_list]
        X = [[g.num_nodes, g.num_edges] for g in gs]
        y = [int(g.y.item()) for g in gs]
        return np.array(X), np.array(y)

    X_train, y_train = features_labels(train_graphs, train_idx)
    X_ext, y_ext = features_labels(ext_graphs)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred_dt = clf.predict(X_ext)

    cm_dt = confusion_matrix(y_true_all, y_pred_dt, labels=[0, 1, 2])
    rows += report_to_rows('decision_tree_size_only', y_true_all, y_pred_dt)

    out_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    acc_gnn = (np.array(y_pred_gnn) == np.array(y_true_all)).mean()
    acc_str = (np.array(y_pred_str_mapped) == np.array(y_true_all)).mean()
    acc_dt = (np.array(y_pred_dt) == np.array(y_true_all)).mean()

    lines = []
    lines.append("=== External dataset (HttpParamsDataset) evaluation -- NO training/fine-tuning ===")
    lines.append(f"n={len(df)} (Benign={sum(1 for y in y_true_all if y==0)}, "
                 f"SQLi={sum(1 for y in y_true_all if y==1)}, XSS={sum(1 for y in y_true_all if y==2)})")
    lines.append("")
    lines.append(f"[1] GATv2 (checkpoint: {MODEL_PATH.name}) overall accuracy: {acc_gnn:.4f}")
    lines.append(classification_report(y_true_all, y_pred_gnn, labels=[0, 1, 2], target_names=CLASS_NAMES, digits=4, zero_division=0))
    lines.append(f"Confusion matrix (rows=true, cols=pred, order Benign/SQLi/XSS):\n{cm_gnn}")
    lines.append("")
    lines.append(f"[2] String-matching baseline (classify_request()) overall accuracy: {acc_str:.4f}")
    lines.append(f"  ({n_unknown_or_other}/{len(df)} rows returned Unknown(-1)/Other(3), counted as wrong)")
    lines.append(classification_report(y_true_all, y_pred_str_mapped, labels=[0, 1, 2], target_names=CLASS_NAMES, digits=4, zero_division=0))
    lines.append(f"Confusion matrix:\n{cm_str}")
    lines.append("")
    lines.append(f"[3] Decision Tree size-only baseline ([num_nodes, num_edges], "
                 f"trained on train_idx n={len(train_idx)}) overall accuracy: {acc_dt:.4f}")
    lines.append(classification_report(y_true_all, y_pred_dt, labels=[0, 1, 2], target_names=CLASS_NAMES, digits=4, zero_division=0))
    lines.append(f"Confusion matrix:\n{cm_dt}")
    lines.append("")
    if acc_dt >= acc_gnn - 0.02:
        lines.append(f"[!] WARNING: Decision Tree size-only accuracy ({acc_dt:.4f}) is close to or exceeds "
                     f"GATv2 accuracy ({acc_gnn:.4f}). This is a signature of the format domain-shift documented "
                     f"in data/external/external_dataset_prep_report.txt (short parameter-value fragments vs. "
                     f"full-request train content) driving classification via graph size alone, not evidence "
                     f"of genuine content-level generalization.")
    else:
        lines.append(f"Decision Tree size-only accuracy ({acc_dt:.4f}) is well below GATv2 ({acc_gnn:.4f}), "
                     f"i.e. graph size alone does not explain GATv2's external performance here.")

    report_text = "\n".join(lines)
    OUT_TXT.write_text(report_text + "\n")
    print(report_text)
    print(f"\nWrote: {OUT_CSV}")
    print(f"Wrote: {OUT_TXT}")


if __name__ == '__main__':
    main()
