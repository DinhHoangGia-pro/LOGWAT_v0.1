"""Graph construction for web token bags."""
import os
import pandas as pd
import torch
import pickle
from torch_geometric.data import Data
from tqdm import tqdm

from src.preprocessing.tokenizer import web_security_tokenizer
from src.bag.node_features import get_node_features
from src.edges.semantic import semantic_edges
from src.utils.config import load_all

cfgs = load_all()
hin_dir = cfgs['dataset'].get('hin_dir')
paths = cfgs['dataset'].get('paths', {})
INPUT_CSV = os.path.join(hin_dir, paths.get('augmented', 'data/augmented_web_attack.csv'))
OUTPUT_GRAPH_FILE = os.path.join(hin_dir, paths.get('graphs_pkl', 'data/web_graphs.pkl'))


def build_web_graphs(input_csv=None, output_file=None):
    input_csv = input_csv or INPUT_CSV
    output_file = output_file or OUTPUT_GRAPH_FILE

    print("--- [1/2] BUILD GRAPH: SEMANTIC DEPENDENCY EDGES ---")
    df = pd.read_csv(input_csv)

    processed_graphs = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Deep Building"):
        tokens = web_security_tokenizer(row.get('content', ''))
        num_nodes = len(tokens)

        x_list = [get_node_features(t, i, num_nodes) for i, t in enumerate(tokens)]

        x = torch.tensor(x_list, dtype=torch.float) if x_list else torch.zeros((1, 64), dtype=torch.float)

        edges = []
        for i in range(num_nodes):
            if i < num_nodes - 1:
                edges.extend([[i, i+1], [i+1, i]])
            if i < num_nodes - 2:
                edges.extend([[i, i+2], [i+2, i]])

        # semantic edges
        sem = semantic_edges(tokens, window=15)
        for (i, j) in sem:
            edges.extend([[i, j]])

        edge_index = torch.tensor(edges if edges else [[0,0]], dtype=torch.long).t().contiguous()
        processed_graphs.append(Data(x=x, edge_index=edge_index, y=torch.tensor([row.get('attack_type', 0)], dtype=torch.long)))

    with open(output_file, 'wb') as f:
        pickle.dump({'graphs': processed_graphs}, f)
    print(f"\n[V] Đã tạo xong đồ thị nâng cấp. Saved to: {output_file}")


if __name__ == '__main__':
    build_web_graphs()
"""Graph builder utilities."""

def build_graph(bag):
    return {'graph': bag}
