"""Build the token-id sequences for the Group-B baselines (Bi-LSTM / TextCNN /
StackLSTM, #16) from the frozen dataset, into data/sequence_baseline/.

Vocabulary is built from the TRAIN rows only (every index not in
test_split_indices.pkl['test_idx'] -- the same rule train.py uses), keeping tokens
with frequency >= MIN_FREQ; ids 0/1 are PAD/UNK. Test/external/held-out tokens
outside it map to UNK. Row i of the output is row i of augmented_web_attack.csv,
so the frozen split indexes it identically to web_graphs.pkl.

Verifies against web_graphs.pkl, for every row: same label, and same node count
(len(token ids) == graph.num_nodes) -- i.e. the sequence baselines see exactly
the tokens the graph baselines' nodes were built from.

Outputs: data/sequence_baseline/web_sequences.pkl  ({'graphs': [Data,...]})
         data/sequence_baseline/vocab.json
"""
import json
import pickle
from collections import Counter

import pandas as pd

from scripts.train_baseline import GRAPHS_PKL
from src.baselines.sequence_data import SEQ_DIR, SEQ_PKL, VOCAB_JSON, build_sequence_data, encode_tokens
from src.models.baselines_seq import PAD_ID, UNK_ID
from src.preprocessing.tokenizer import web_security_tokenizer

ROOT = SEQ_DIR.parents[1]
MIN_FREQ = 2


def main():
    df = pd.read_csv(ROOT / 'data' / 'augmented_web_attack.csv')
    with open(ROOT / 'data' / 'test_split_indices.pkl', 'rb') as f:
        test_set = set(pickle.load(f)['test_idx'])
    train_idx = [i for i in range(len(df)) if i not in test_set]

    contents = df['content'].astype(str).tolist()
    freq = Counter(t for i in train_idx for t in web_security_tokenizer(contents[i]))
    kept = sorted(t for t, c in freq.items() if c >= MIN_FREQ)
    vocab = {t: i + 2 for i, t in enumerate(kept)}
    print(f"[*] train rows={len(train_idx)} | distinct train tokens={len(freq)} | kept (freq>={MIN_FREQ})={len(kept)} "
          f"| vocab_size incl. PAD/UNK={len(vocab) + 2}")

    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    with open(VOCAB_JSON, 'w') as f:
        json.dump({'min_freq': MIN_FREQ, 'pad_id': PAD_ID, 'unk_id': UNK_ID,
                   'vocab_size': len(vocab) + 2, 'token_to_id': vocab}, f)

    seqs = [build_sequence_data(contents[i], vocab, source_uid=df.loc[i, 'source_uid'],
                                attack_type=df.loc[i, 'attack_type'])
            for i in range(len(df))]
    with open(SEQ_PKL, 'wb') as f:
        pickle.dump({'graphs': seqs}, f)

    with open(GRAPHS_PKL, 'rb') as f:
        graphs = pickle.load(f)['graphs']
    assert len(graphs) == len(seqs)
    for i, (g, s) in enumerate(zip(graphs, seqs)):
        assert int(g.y) == int(s.y), f"label differs at row {i}"
        assert g.num_nodes == s.x.numel(), f"node/token count differs at row {i}: {g.num_nodes} vs {s.x.numel()}"
    unk_test = sum((seqs[i].x == UNK_ID).sum().item() for i in test_set)
    tot_test = sum(seqs[i].x.numel() for i in test_set)
    print(f"[V] verified {len(seqs)} rows: labels and per-row node counts match web_graphs.pkl")
    print(f"[*] test-split UNK rate = {unk_test}/{tot_test} = {unk_test / tot_test:.4f}")
    print(f"[*] wrote {SEQ_PKL} and {VOCAB_JSON}")


if __name__ == '__main__':
    main()
