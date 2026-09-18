"""5-seed stability check for the E_sem window-boundary held-out test
(`scripts/build_heldout_window_test.py`'s attribute_spacing_in_range
gap=12 and attribute_spacing_out_of_range gap=34 cells).

Reuses the 5 checkpoints from the 5-seed run behind
`results/final_stats_5seed.csv` / `results/heldout_percell_5seed.csv`
(no retraining):
  seed 42 -> data/models_pretrained/best_web_gnn_seed42_5seed_run.pth
             (the standalone rerun used to recover the seed=42 per-cell
             result after the original loop iteration's checkpoint was
             overwritten by seed=43's run -- see EXPERIMENT_LOG "5-seed
             statistics" and this script's own docstring precedent in
             run_5seed_stats.py)
  seed 43-46 -> data/models_pretrained/best_web_gnn_seed{N}.pth

Reads the existing, already-frozen `data/heldout_window_test.csv` (does
NOT regenerate it -- that file's content is untouched by this script).

Writes:
  results/gap_window_5seed_raw.csv    per-row: seed, group, source_uid,
                                       true_label, pred_label, correct,
                                       prob_benign, prob_sqli, prob_xss,
                                       logit_gap (prob[true] - max(prob[other]))
  results/gap_window_5seed.csv        per-seed/group aggregate: seed, group,
                                       accuracy, n_correct, n_total,
                                       mean_logit_gap
"""
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.bag.graph_builder import build_single_graph
from src.models.logwat import HeavyWebGNN

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
HELDOUT_CSV = DATA_DIR / 'heldout_window_test.csv'
RAW_CSV = RESULTS_DIR / 'gap_window_5seed_raw.csv'
SUMMARY_CSV = RESULTS_DIR / 'gap_window_5seed.csv'

CHECKPOINTS = {
    42: DATA_DIR / 'models_pretrained' / 'best_web_gnn_seed42_5seed_run.pth',
    43: DATA_DIR / 'models_pretrained' / 'best_web_gnn_seed43.pth',
    44: DATA_DIR / 'models_pretrained' / 'best_web_gnn_seed44.pth',
    45: DATA_DIR / 'models_pretrained' / 'best_web_gnn_seed45.pth',
    46: DATA_DIR / 'models_pretrained' / 'best_web_gnn_seed46.pth',
}

CLASS_NAMES = ['benign', 'sqli', 'xss']


def evaluate_seed(seed, model_path, df):
    model = HeavyWebGNN(use_edge_attr=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    graphs = [build_single_graph(str(r['content']), source_uid=str(r['source_uid']),
                                  attack_type=int(r['attack_type']),
                                  use_seq=True, use_skip=True, use_sem=True, use_edge_attr=False)
              for _, r in df.iterrows()]

    probs_all = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=8, shuffle=False):
            logits = model(batch.x, batch.edge_index, batch.batch, edge_attr=None)
            probs_all.append(F.softmax(logits, dim=1))
    probs = torch.cat(probs_all, dim=0)

    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        true_label = int(r['attack_type'])
        row_probs = probs[i].tolist()
        pred_label = int(probs[i].argmax().item())
        other_max = max(p for c, p in enumerate(row_probs) if c != true_label)
        rows.append({
            'seed': seed, 'group': r['technique'], 'source_uid': r['source_uid'],
            'true_label': true_label, 'pred_label': pred_label,
            'correct': int(pred_label == true_label),
            'prob_benign': row_probs[0], 'prob_sqli': row_probs[1], 'prob_xss': row_probs[2],
            'logit_gap': row_probs[true_label] - other_max,
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(HELDOUT_CSV)
    assert set(df['technique'].unique()) == {'attribute_spacing_in_range', 'attribute_spacing_out_of_range'}
    assert len(df) == 20, f"expected 20 rows (10/group), got {len(df)}"

    for seed, path in CHECKPOINTS.items():
        if not path.exists():
            raise FileNotFoundError(f"seed {seed} checkpoint missing: {path}")

    raw_dfs = []
    for seed, path in CHECKPOINTS.items():
        print(f"[*] seed={seed} model={path}")
        raw_dfs.append(evaluate_seed(seed, path, df))
    raw_df = pd.concat(raw_dfs, ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(RAW_CSV, index=False)

    summary_rows = []
    for seed in CHECKPOINTS:
        for group in ['attribute_spacing_in_range', 'attribute_spacing_out_of_range']:
            subset = raw_df[(raw_df['seed'] == seed) & (raw_df['group'] == group)]
            summary_rows.append({
                'seed': seed, 'group': group,
                'accuracy': subset['correct'].mean(),
                'n_correct': int(subset['correct'].sum()),
                'n_total': len(subset),
                'mean_logit_gap': subset['logit_gap'].mean(),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("\n" + summary_df.to_string(index=False))
    print(f"\nWrote: {RAW_CSV}")
    print(f"Wrote: {SUMMARY_CSV}")


if __name__ == '__main__':
    main()
