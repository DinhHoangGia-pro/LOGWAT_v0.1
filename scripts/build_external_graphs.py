"""Build BAG graphs for the external evaluation dataset only.

Explicit use_seq/use_skip/use_sem/use_edge_attr, matching the deployed
checkpoint's configuration exactly (verified against
best_web_gnn_seed42_ablation_1_full_no_edge_attr.pth in
docs/REPRODUCIBILITY.md's Data Integrity / checkpoint history: full edges,
no edge_attr). Deliberately NOT relying on build_web_graphs()'s current
defaults, since those could silently drift after future ablation work.
"""
from pathlib import Path

from src.bag.graph_builder import build_web_graphs

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = ROOT / 'data' / 'external'
INPUT_CSV = EXTERNAL_DIR / 'external_dataset_clean.csv'
OUTPUT_PKL = EXTERNAL_DIR / 'external_dataset_graphs.pkl'


def main():
    build_web_graphs(
        input_csv=str(INPUT_CSV),
        output_file=str(OUTPUT_PKL),
        use_seq=True,
        use_skip=True,
        use_sem=True,
        use_edge_attr=False,
    )


if __name__ == '__main__':
    main()
