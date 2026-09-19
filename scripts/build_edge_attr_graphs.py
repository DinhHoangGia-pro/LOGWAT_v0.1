"""Build the edge_attr (relation one-hot) variant of the training graphs, and of
the external-dataset graphs, into SEPARATE .pkl files, for HGT (#16 Group A).

HGT is the only baseline that needs edge types. web_graphs.pkl (frozen
pipeline state, use_edge_attr=False) is deliberately not overwritten: these
files sit beside it, and are checked (see --verify) to hold graphs whose
x / edge_index / y are identical to the originals -- only `edge_attr` is
added -- so HGT is trained on exactly the same graph structure as everything
else.

  data/web_graphs.pkl            -> data/web_graphs_edge_attr.pkl
  data/external/external_dataset_graphs.pkl
                                 -> data/external/external_dataset_graphs_edge_attr.pkl
"""
import argparse
import pickle
from pathlib import Path

import torch

from src.bag.graph_builder import build_web_graphs

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'

JOBS = {
    'train': (DATA_DIR / 'augmented_web_attack.csv', DATA_DIR / 'web_graphs.pkl',
              DATA_DIR / 'web_graphs_edge_attr.pkl'),
    'external': (DATA_DIR / 'external' / 'external_dataset_clean.csv',
                 DATA_DIR / 'external' / 'external_dataset_graphs.pkl',
                 DATA_DIR / 'external' / 'external_dataset_graphs_edge_attr.pkl'),
}


def verify(orig_pkl, new_pkl):
    with open(orig_pkl, 'rb') as f:
        orig = pickle.load(f)['graphs']
    with open(new_pkl, 'rb') as f:
        new = pickle.load(f)['graphs']
    assert len(orig) == len(new), f"length mismatch {len(orig)} vs {len(new)}"
    for i, (a, b) in enumerate(zip(orig, new)):
        assert torch.equal(a.x, b.x), f"x differs at graph {i}"
        assert torch.equal(a.edge_index, b.edge_index), f"edge_index differs at graph {i}"
        assert torch.equal(a.y, b.y), f"y differs at graph {i}"
        assert str(a.source_uid) == str(b.source_uid), f"source_uid differs at graph {i}"
        assert b.edge_attr.shape == (b.edge_index.size(1), 3), f"bad edge_attr shape at graph {i}"
    print(f"[V] verified {len(orig)} graphs: x/edge_index/y/source_uid identical to {orig_pkl.name}, edge_attr added")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--which', nargs='*', default=list(JOBS))
    args = parser.parse_args()
    for name in args.which:
        csv, orig, out = JOBS[name]
        build_web_graphs(input_csv=str(csv), output_file=str(out),
                         use_seq=True, use_skip=True, use_sem=True, use_edge_attr=True)
        verify(orig, out)


if __name__ == '__main__':
    main()
