"""Graph construction for web token bags."""
import os
import pandas as pd
import torch
import pickle
from torch_geometric.data import Data
from tqdm import tqdm

from src.preprocessing.tokenizer import web_security_tokenizer
from src.bag.node_features import get_node_features
from src.edges.sequential import sequential_edges
from src.edges.skip import skip_edges
from src.edges.semantic import semantic_edges
from src.utils.config import load_all

cfgs = load_all()
hin_dir = cfgs['dataset'].get('hin_dir')
paths = cfgs['dataset'].get('paths', {})
INPUT_CSV = os.path.join(hin_dir, paths.get('augmented', 'data/augmented_web_attack.csv'))
OUTPUT_GRAPH_FILE = os.path.join(hin_dir, paths.get('graphs_pkl', 'data/web_graphs.pkl'))

# One-hot relation-type columns for edge_attr, in a fixed order.
RELATION_TYPES = ('seq', 'skip', 'sem')


def build_single_graph(content, source_uid=None, attack_type=0,
                        use_seq=True, use_skip=True, use_sem=True,
                        use_edge_attr=False, sem_window=15):
    """Build one `torch_geometric.data.Data` graph from raw `content`.

    Shared by `build_web_graphs()` and any script that needs to build a
    single ad-hoc graph (held-out eval, diagnostic probes) so all callers
    stay consistent with the same seq/skip/sem toggles and edge_attr
    encoding instead of re-implementing this logic per script.
    """
    tokens = web_security_tokenizer(content)
    num_nodes = len(tokens)

    x_list = [get_node_features(t, i, num_nodes) for i, t in enumerate(tokens)]
    x = torch.tensor(x_list, dtype=torch.float) if x_list else torch.zeros((1, 64), dtype=torch.float)

    edges = []
    relations = []  # parallel list of 'seq'/'skip'/'sem', one per edge in `edges`

    if use_seq:
        seq = sequential_edges(tokens)
        edges.extend(seq)
        relations.extend(['seq'] * len(seq))
    if use_skip:
        skip = skip_edges(tokens, k=2)
        edges.extend(skip)
        relations.extend(['skip'] * len(skip))
    if use_sem:
        sem = semantic_edges(tokens, window=sem_window)
        edges.extend(sem)
        relations.extend(['sem'] * len(sem))

    if not edges:
        edges = [(0, 0)]
        relations = ['seq']

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    kwargs = {}
    if use_edge_attr:
        rel_to_idx = {r: i for i, r in enumerate(RELATION_TYPES)}
        edge_attr = torch.zeros((len(relations), len(RELATION_TYPES)), dtype=torch.float)
        for e, r in enumerate(relations):
            edge_attr[e, rel_to_idx[r]] = 1.0
        kwargs['edge_attr'] = edge_attr

    return Data(x=x, edge_index=edge_index,
                y=torch.tensor([attack_type], dtype=torch.long),
                source_uid=source_uid, **kwargs)


def build_web_graphs(input_csv=None, output_file=None,
                      use_seq=True, use_skip=True, use_sem=True,
                      use_edge_attr=False):
    input_csv = input_csv or INPUT_CSV
    output_file = output_file or OUTPUT_GRAPH_FILE

    print("--- [1/2] BUILD GRAPH: SEMANTIC DEPENDENCY EDGES ---")
    print(f"[*] use_seq={use_seq} use_skip={use_skip} use_sem={use_sem} use_edge_attr={use_edge_attr}")
    df = pd.read_csv(input_csv)

    processed_graphs = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Deep Building"):
        g = build_single_graph(
            row.get('content', ''),
            source_uid=row.get('source_uid', None),
            attack_type=row.get('attack_type', 0),
            use_seq=use_seq, use_skip=use_skip, use_sem=use_sem,
            use_edge_attr=use_edge_attr,
        )
        processed_graphs.append(g)

    with open(output_file, 'wb') as f:
        pickle.dump({'graphs': processed_graphs}, f)
    print(f"\n[V] Đã tạo xong đồ thị nâng cấp. Saved to: {output_file}")


if __name__ == '__main__':
    build_web_graphs()
