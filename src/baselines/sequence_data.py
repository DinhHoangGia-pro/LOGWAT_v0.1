"""Token-id encoding shared by the sequence baselines' data prep and evaluation.

A sequence "graph" is a `torch_geometric.data.Data` whose `.x` is a LongTensor of
token ids (one per token, in order) and whose `edge_index` is empty. Encoding
requests this way lets the sequence baselines reuse the GATv2 training/eval
DataLoader path unchanged (see src/models/baselines_seq.py).
"""
import json
from pathlib import Path

import torch
from torch_geometric.data import Data

from src.models.baselines_seq import PAD_ID, UNK_ID
from src.preprocessing.tokenizer import web_security_tokenizer

ROOT = Path(__file__).resolve().parents[2]
SEQ_DIR = ROOT / 'data' / 'sequence_baseline'
SEQ_PKL = SEQ_DIR / 'web_sequences.pkl'
VOCAB_JSON = SEQ_DIR / 'vocab.json'


def load_vocab(path=VOCAB_JSON):
    with open(path) as f:
        return json.load(f)['token_to_id']


def encode_tokens(text, vocab):
    """ids for `web_security_tokenizer(text)`; an empty token list becomes a single
    UNK so every request has >=1 node (mirrors build_single_graph's 1-node fallback)."""
    ids = [vocab.get(t, UNK_ID) for t in web_security_tokenizer(text)]
    return ids or [UNK_ID]


def build_sequence_data(text, vocab, source_uid=None, attack_type=0):
    ids = torch.tensor(encode_tokens(text, vocab), dtype=torch.long)
    return Data(x=ids, edge_index=torch.empty((2, 0), dtype=torch.long),
                y=torch.tensor([int(attack_type)], dtype=torch.long), source_uid=source_uid)
