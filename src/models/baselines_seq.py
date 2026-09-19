"""Sequence-based baselines (Group B of the baseline-fairness rerun, #16).

These read the RAW token sequence from `web_security_tokenizer(content)` -- the
same tokens the BAG is built from -- through an embedding layer learned from
scratch (no pretrained vectors, matching GATv2 which also learns from scratch;
unlike RoBERTa/CodeBERT, which are pretrained). They see NO edges and none of
the hand-crafted per-token node features (danger-char / SQL-keyword flags):
just token ids, so any gap to the graph models reflects both the missing
graph structure and the learned-vs-engineered token representation.

Interface is the same `forward(x, edge_index, batch, edge_attr=None)` as the graph
models so `src.training.train.train()` runs them unchanged. `x` is a LongTensor of
token ids, one per node, in token order (nodes of a batched graph are concatenated,
`batch` says which request each belongs to); `to_dense_batch` re-pads them into
[B, L]. `edge_index` / `edge_attr` are ignored.

Token id 0 = PAD, 1 = UNK (any token seen fewer than `min_freq` times in TRAIN, or
never -- this is what a from-scratch vocabulary does with unseen tokens, and it is
part of what the held-out/external evaluations measure).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.utils import to_dense_batch

from src.models.layers import DEFAULT_HIDDEN, DEFAULT_OUT

PAD_ID = 0
UNK_ID = 1
EMB_DIM = 128


def _head(in_dim, hidden_dim, out_channels):
    """Same MLP head shape as layers.Classifier's inner Sequential."""
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                         nn.Linear(hidden_dim, out_channels))


class _SeqBase(nn.Module):
    needs_edge_attr = False

    def __init__(self, vocab_size, emb_dim=EMB_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)

    def _dense(self, x, batch, min_len=1):
        ids, mask = to_dense_batch(x, batch, fill_value=PAD_ID)
        if ids.size(1) < min_len:
            pad = min_len - ids.size(1)
            ids = F.pad(ids, (0, pad), value=PAD_ID)
            mask = F.pad(mask, (0, pad), value=False)
        return ids, mask


class BiLSTMBaseline(_SeqBase):
    """embedding -> 1-layer BiLSTM -> [masked mean-pool ; masked max-pool] -> MLP head."""

    def __init__(self, vocab_size, emb_dim=EMB_DIM, hidden_dim=DEFAULT_HIDDEN, out_channels=DEFAULT_OUT):
        super().__init__(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = _head(hidden_dim * 2, hidden_dim, out_channels)

    def forward(self, x, edge_index, batch, edge_attr=None):
        ids, mask = self._dense(x, batch)
        lengths = mask.sum(1).cpu()
        packed = pack_padded_sequence(self.embedding(ids), lengths, batch_first=True, enforce_sorted=False)
        out, _ = pad_packed_sequence(self.lstm(packed)[0], batch_first=True, total_length=ids.size(1))
        m = mask.unsqueeze(-1)
        mean = (out * m).sum(1) / m.sum(1).clamp(min=1)
        mx = out.masked_fill(~m, float('-inf')).max(1).values
        return self.classifier(torch.cat([mean, mx], dim=1))


class TextCNNBaseline(_SeqBase):
    """Kim (2014): embedding -> Conv1d(kernels 3,4,5) -> ReLU -> max-over-time -> dropout -> linear."""

    KERNELS = (3, 4, 5)

    def __init__(self, vocab_size, emb_dim=EMB_DIM, num_filters=100, out_channels=DEFAULT_OUT):
        super().__init__(vocab_size, emb_dim)
        self.convs = nn.ModuleList(nn.Conv1d(emb_dim, num_filters, k) for k in self.KERNELS)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(self.KERNELS), out_channels)

    def forward(self, x, edge_index, batch, edge_attr=None):
        ids, mask = self._dense(x, batch, min_len=max(self.KERNELS))
        lengths = mask.sum(1)
        e = self.embedding(ids).transpose(1, 2)  # [B, E, L]
        pooled = []
        for conv in self.convs:
            k = conv.kernel_size[0]
            h = F.relu(conv(e))  # [B, F, L-k+1]
            # only windows fully inside the real sequence count; a request shorter
            # than k keeps its first (padded) window so max-over-time is defined
            n_valid = (lengths - k + 1).clamp(min=1)
            valid = torch.arange(h.size(2), device=h.device).unsqueeze(0) < n_valid.unsqueeze(1)
            pooled.append(h.masked_fill(~valid.unsqueeze(1), float('-inf')).max(2).values)
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class StackLSTMBaseline(_SeqBase):
    """embedding -> 2 stacked unidirectional LSTM layers -> last real hidden state -> MLP head."""

    def __init__(self, vocab_size, emb_dim=EMB_DIM, hidden_dim=DEFAULT_HIDDEN, out_channels=DEFAULT_OUT):
        super().__init__(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=False)
        self.classifier = _head(hidden_dim, hidden_dim, out_channels)

    def forward(self, x, edge_index, batch, edge_attr=None):
        ids, mask = self._dense(x, batch)
        lengths = mask.sum(1).cpu()
        packed = pack_padded_sequence(self.embedding(ids), lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.classifier(h_n[-1])


SEQ_BASELINES = {
    'bilstm': BiLSTMBaseline,
    'textcnn': TextCNNBaseline,
    'stacklstm': StackLSTMBaseline,
}
