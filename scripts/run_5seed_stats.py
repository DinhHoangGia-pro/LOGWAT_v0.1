"""Train the OFFICIAL config (full edges: E_seq+E_skip+E_sem, use_edge_attr=False,
current dataset = Mechanism 1 + 2 + 2b, 43659 rows) across 5 seeds
(42, 43, 44, 45, 46) to get mean+-std statistics instead of a single-seed
point estimate.

For each seed: backs up the resulting checkpoint/log to
`best_web_gnn_seed{N}.pth` / `training_history_seed{N}.log`, evaluates on
BOTH the frozen test split and the 9-cell held-out matrix, and records
per-seed metrics.

Does NOT restore the deployed `best_web_gnn_seed42.pth`/logs/results itself
-- the caller must back those up before running this script and restore
them after (same discipline as the ablation scripts), since this script's
own last loop iteration (seed=46) is what ends up left in
`best_web_gnn_seed42.pth` when it exits (`train()` always writes to that
fixed path regardless of `seed`; this script only copies each seed's result
to a `best_web_gnn_seed{N}.pth` sibling right after that seed's run).

Writes `results/final_stats_5seed.csv`.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.loader import DataLoader

from src.models.logwat import HeavyWebGNN
from src.training import train as train_mod
from scripts.build_heldout_matrix_eval import make_matrix_rows, evaluate_group

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
LOGS_DIR = ROOT / 'logs'
MODEL_PATH = DATA_DIR / 'models_pretrained' / 'best_web_gnn_seed42.pth'
LOG_PATH = LOGS_DIR / 'training_history.log'
MAIN_RESULTS_PATH = RESULTS_DIR / 'main_results.csv'
HELDOUT_RESULTS_PATH = RESULTS_DIR / 'heldout_matrix_full.csv'
STATS_CSV = RESULTS_DIR / 'final_stats_5seed.csv'
GRAPHS_PKL = DATA_DIR / 'web_graphs.pkl'
TEST_SPLIT_PKL = DATA_DIR / 'test_split_indices.pkl'

SEEDS = [42, 43, 44, 45, 46]
BASELINE_HGT_ACC = 0.9587  # Table 2, strongest external baseline
CLASS_NAMES = ['Benign', 'SQLi', 'XSS']


def evaluate_test_split(model_path):
    import pickle
    with open(GRAPHS_PKL, 'rb') as f:
        graphs = pickle.load(f)['graphs']
    with open(TEST_SPLIT_PKL, 'rb') as f:
        split = pickle.load(f)
    test_idx = split['test_idx']

    model = HeavyWebGNN(use_edge_attr=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    test_graphs = [graphs[i] for i in test_idx]
    y_true = [int(g.y.item()) for g in test_graphs]
    y_pred = []
    with torch.no_grad():
        for batch in DataLoader(test_graphs, batch_size=256, shuffle=False):
            out = model(batch.x, batch.edge_index, batch.batch, edge_attr=None)
            y_pred.extend(int(p) for p in out.argmax(dim=1).tolist())

    report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
                                    output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return report, cm


def main():
    per_seed_rows = []
    heldout_df = make_matrix_rows()

    for seed in SEEDS:
        print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")
        train_mod.train(seed=seed)

        seed_model_path = DATA_DIR / 'models_pretrained' / f'best_web_gnn_seed{seed}.pth'
        seed_log_path = LOGS_DIR / f'training_history_seed{seed}.log'
        if seed_model_path != MODEL_PATH:
            shutil.copy(MODEL_PATH, seed_model_path)
        if seed_log_path != LOG_PATH:
            shutil.copy(LOG_PATH, seed_log_path)

        report, cm = evaluate_test_split(seed_model_path)
        heldout_result = evaluate_group(heldout_df, model_path=str(seed_model_path),
                                         use_seq=True, use_skip=True, use_sem=True, use_edge_attr=False)
        cells_correct = int((heldout_result['gatv2_accuracy'] == 1.0).sum())

        row = {'seed': seed, 'accuracy': report['accuracy'],
               'macro_precision': report['macro avg']['precision'],
               'macro_recall': report['macro avg']['recall'],
               'macro_f1': report['macro avg']['f1-score'],
               'heldout_cells_correct': cells_correct, 'heldout_cells_total': 9}
        for cls in CLASS_NAMES:
            row[f'{cls.lower()}_precision'] = report[cls]['precision']
            row[f'{cls.lower()}_recall'] = report[cls]['recall']
            row[f'{cls.lower()}_f1'] = report[cls]['f1-score']
        per_seed_rows.append(row)
        print(f"seed={seed}: acc={report['accuracy']:.6f} macro_f1={report['macro avg']['f1-score']:.6f} "
              f"heldout={cells_correct}/9")
        print(f"  confusion matrix:\n{cm}")

    per_seed_df = pd.DataFrame(per_seed_rows)

    metric_cols = [c for c in per_seed_df.columns if c not in ('seed', 'heldout_cells_correct', 'heldout_cells_total')]
    summary_rows = []
    for col in metric_cols:
        vals = per_seed_df[col].values
        summary_rows.append({'metric': col, 'mean': vals.mean(), 'std': vals.std(ddof=1),
                             'min': vals.min(), 'max': vals.max()})

    acc_mean = per_seed_df['accuracy'].mean()
    acc_std = per_seed_df['accuracy'].std(ddof=1)
    cohens_d = (acc_mean - BASELINE_HGT_ACC) / acc_std if acc_std > 0 else float('inf')
    summary_rows.append({'metric': 'cohens_d_vs_HGT_95.87pct', 'mean': cohens_d, 'std': '',
                         'min': '', 'max': ''})

    heldout_vals = per_seed_df['heldout_cells_correct'].values
    summary_rows.append({'metric': 'heldout_cells_correct_/9', 'mean': heldout_vals.mean(),
                         'std': heldout_vals.std(ddof=1), 'min': heldout_vals.min(), 'max': heldout_vals.max()})

    summary_df = pd.DataFrame(summary_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATS_CSV, 'w') as f:
        f.write("=== per-seed results ===\n")
        per_seed_df.to_csv(f, index=False)
        f.write("\n=== mean/std/min/max across 5 seeds ===\n")
        summary_df.to_csv(f, index=False)

    print("\n" + "=" * 70)
    print("PER-SEED RESULTS")
    print(per_seed_df.to_string(index=False))
    print("\nSUMMARY (mean +/- std)")
    print(summary_df.to_string(index=False))
    print(f"\nWrote: {STATS_CSV}")
    print(f"\n[!] best_web_gnn_seed42.pth now holds the LAST loop iteration's checkpoint "
          f"(seed={SEEDS[-1]}), not the originally-deployed one. Caller must restore the "
          f"deployed checkpoint/log/results from its own backup.")


if __name__ == '__main__':
    main()
