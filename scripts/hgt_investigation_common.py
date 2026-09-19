"""Shared helpers for the HGT investigation scripts (edge-type use, generalization, external).

`load_model(key, seed)` keys:
  hgt            typed HGT baseline           (needs edge_attr)          data: *_edge_attr.pkl
  hgt_collapsed  one-edge-type HGT control    (ignores edge_attr)
  hgt_random     random-edge-type HGT control (ignores edge_attr)
  gatv2          deployed LOGWAT GATv2 (no edge_attr)

`apply_condition(g, cond)` returns a copy of an edge_attr-carrying graph with the relation
information of its edges edited, so the SAME trained checkpoint can be probed causally.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from scripts.build_heldout_matrix_eval import LABELS, TECHNIQUES, make_matrix_rows
from scripts.train_baseline import baseline_spec, ckpt_path as base_ckpt_path
from scripts.train_hgt_control import ckpt_path as ctl_ckpt_path
from src.bag.graph_builder import RELATION_TYPES, build_single_graph
from src.models.hgt_controls import CONTROL_NAMES, HGTControl

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
SEEDS = [42, 43, 44, 45, 46]
CLASS_NAMES = ['Benign', 'SQLi', 'XSS']
GRAPHS_EA = DATA_DIR / 'web_graphs_edge_attr.pkl'
EXTERNAL_CSV = DATA_DIR / 'external' / 'external_dataset_clean.csv'
EXTERNAL_EA = DATA_DIR / 'external' / 'external_dataset_graphs_edge_attr.pkl'
SPLIT_PKL = DATA_DIR / 'test_split_indices.pkl'
SEQ, SKIP, SEM = range(3)


def load_model(key, seed, device='cpu'):
    if key in CONTROL_NAMES:
        model = HGTControl(CONTROL_NAMES[key])
        path = ctl_ckpt_path(key, seed)
    else:
        factory, _ = baseline_spec(key)
        model = factory(False)
        path = base_ckpt_path(key, seed)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def uses_edge_attr(key):
    return key == 'hgt'


def _one_hot(rel):
    ea = torch.zeros((len(rel), len(RELATION_TYPES)))
    ea[torch.arange(len(rel)), rel] = 1.0
    return ea


def _hash_rel(edge_index):
    return ((edge_index[0] * 73856093) ^ (edge_index[1] * 19349663)) % 3


def apply_condition(g, cond):
    """cond in: real | all_seq | all_skip | all_sem | cyclic | swap_seq_skip | sem_as_seq | sem_as_skip |
    random | drop_sem | drop_skip | drop_seq.   Graph must carry edge_attr (one-hot seq/skip/sem)."""
    rel = g.edge_attr.argmax(dim=1)
    ei = g.edge_index
    if cond == 'real':
        new_rel = rel
    elif cond.startswith('all_'):
        new_rel = torch.full_like(rel, RELATION_TYPES.index(cond[4:]))
    elif cond == 'cyclic':
        new_rel = (rel + 1) % 3
    elif cond == 'swap_seq_skip':
        new_rel = torch.where(rel == SEQ, torch.full_like(rel, SKIP), torch.where(rel == SKIP, torch.full_like(rel, SEQ), rel))
    elif cond == 'sem_as_seq':
        new_rel = torch.where(rel == SEM, torch.full_like(rel, SEQ), rel)
    elif cond == 'sem_as_skip':
        new_rel = torch.where(rel == SEM, torch.full_like(rel, SKIP), rel)
    elif cond == 'random':
        new_rel = _hash_rel(ei)
    elif cond.startswith('drop_'):
        keep = rel != RELATION_TYPES.index(cond[5:])
        if not keep.any():                      # never hand a model an edgeless graph
            keep = torch.ones_like(keep)
        ei, new_rel = ei[:, keep], rel[keep]
    else:
        raise ValueError(cond)
    return Data(x=g.x, edge_index=ei, edge_attr=_one_hot(new_rel), y=g.y, source_uid=g.source_uid)


@torch.no_grad()
def logits_of(model, graphs, use_ea, device='cpu', batch_size=256):
    out = []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        ea = batch.edge_attr if use_ea else None
        out.append(model(batch.x, batch.edge_index, batch.batch, edge_attr=ea).cpu())
    return torch.cat(out)


def matrix_graphs():
    """-> (df, graphs with edge_attr) for the exact 90 held-out rows."""
    df = make_matrix_rows()
    graphs = [build_single_graph(str(r.content), source_uid=str(r.source_uid), attack_type=int(r.attack_type),
                                 use_edge_attr=True) for r in df.itertuples()]
    return df, graphs


def cell_correct(df, preds):
    """-> {'Class/technique': n_correct} in matrix order."""
    preds = np.asarray(preds)
    res = {}
    for label in (0, 1, 2):
        for tech in TECHNIQUES[label]:
            m = ((df['class_name'] == LABELS[label]) & (df['technique'] == tech)).values
            res[f"{LABELS[label]}/{tech}"] = int((preds[m] == df['attack_type'].values[m]).sum())
    return res


def load_pkl_graphs(path):
    with open(path, 'rb') as f:
        return pickle.load(f)['graphs']


def test_split_graphs():
    graphs = load_pkl_graphs(GRAPHS_EA)
    with open(SPLIT_PKL, 'rb') as f:
        test_idx = list(pickle.load(f)['test_idx'])
    return [graphs[i] for i in test_idx]


def margin(logits, cls):
    """logit[cls] - max(other logits): >0 means cls wins."""
    other = logits.clone()
    other[:, cls] = -1e9
    return logits[:, cls] - other.max(dim=1).values
