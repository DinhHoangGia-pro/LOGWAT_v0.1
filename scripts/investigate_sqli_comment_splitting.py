import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.bag.node_features import get_node_features
from src.edges.semantic import semantic_edges
from src.models.logwat import HeavyWebGNN
from src.preprocessing.tokenizer import web_security_tokenizer

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
HELDOUT_CSV = DATA_DIR / 'heldout_keyword_test.csv'
MODEL_PATH = DATA_DIR / 'models_pretrained' / 'best_web_gnn_seed42.pth'

LABELS = {0: 'Benign', 1: 'SQLi', 2: 'XSS'}
KNOWN_MEANS = {'Benign': 41.2, 'SQLi': 10.7, 'XSS': 21.7}


def build_graph(content: str, attack_type: int, source_uid: str) -> Data:
    tokens = web_security_tokenizer(content)
    x_list = [get_node_features(t, i, len(tokens) or 1) for i, t in enumerate(tokens)]
    x = torch.tensor(x_list, dtype=torch.float) if x_list else torch.zeros((1, 64), dtype=torch.float)
    edges = []
    for i in range(len(tokens)):
        if i < len(tokens) - 1:
            edges.extend([[i, i + 1], [i + 1, i]])
        if i < len(tokens) - 2:
            edges.extend([[i, i + 2], [i + 2, i]])
    for (i, j) in semantic_edges(tokens, window=15):
        edges.append([i, j])
    edge_index = torch.tensor(edges if edges else [[0, 0]], dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index, y=torch.tensor([attack_type], dtype=torch.long), source_uid=source_uid)


def main():
    df = pd.read_csv(HELDOUT_CSV)
    subset = df[(df['class_name'] == 'SQLi') & (df['technique'] == 'comment_splitting')].copy()
    assert len(subset) == 10, f"expected 10 rows, got {len(subset)}"

    model = HeavyWebGNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    graphs = []
    for _, row in subset.iterrows():
        graphs.append(build_graph(str(row['content']), int(row['attack_type']), str(row['source_uid'])))

    y_pred = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=8, shuffle=False):
            logits = model(batch.x, batch.edge_index, batch.batch)
            y_pred.extend(int(p) for p in logits.argmax(dim=1).tolist())

    subset['pred'] = y_pred
    subset['pred_name'] = subset['pred'].map(LABELS)
    subset['num_nodes'] = [g.num_nodes for g in graphs]
    subset['num_edges'] = [g.num_edges for g in graphs]

    # ---- Task 1: raw samples + predicted labels ----
    lines1 = []
    lines1.append("=" * 90)
    lines1.append("SQLi comment_splitting: 10 mau that + nhan du doan cua model")
    lines1.append("=" * 90)
    for _, row in subset.iterrows():
        lines1.append(
            f"[{row['source_uid']}] true=SQLi pred={row['pred_name']} "
            f"nodes={row['num_nodes']} edges={row['num_edges']} content={row['content']!r}"
        )
    print("\n".join(lines1))
    with open(DATA_DIR / 'sqli_comment_splitting_samples.txt', 'w') as f:
        f.write("\n".join(lines1) + "\n")

    # ---- Task 2: graph sizes vs known means ----
    lines2 = []
    lines2.append("=" * 90)
    lines2.append("SQLi comment_splitting: num_nodes/num_edges cua 10 graph,")
    lines2.append("so sanh voi mean SQLi=10.7 va mean Benign=41.2")
    lines2.append("=" * 90)
    for _, row in subset.iterrows():
        lines2.append(
            f"[{row['source_uid']}] num_nodes={row['num_nodes']} num_edges={row['num_edges']} pred={row['pred_name']}"
        )
    mean_nodes = subset['num_nodes'].mean()
    mean_edges = subset['num_edges'].mean()
    lines2.append(f"\nmean num_nodes={mean_nodes:.1f}, mean num_edges={mean_edges:.1f}")
    lines2.append(f"so voi known means: Benign=41.2, SQLi=10.7, XSS=21.7")
    closest = min(KNOWN_MEANS.items(), key=lambda kv: abs(kv[1] - mean_nodes))
    lines2.append(f"gan nhat voi lop: {closest[0]} (mean={closest[1]})")
    print("\n" + "\n".join(lines2))
    with open(DATA_DIR / 'sqli_comment_splitting_graph_sizes.txt', 'w') as f:
        f.write("\n".join(lines2) + "\n")

    # ---- Task 3: distribution of predicted classes for these 10 samples ----
    lines3 = []
    lines3.append("=" * 90)
    lines3.append("SQLi comment_splitting: phan bo lop du doan cho 10 mau nay")
    lines3.append("=" * 90)
    pred_counts = subset['pred_name'].value_counts().reindex(['Benign', 'SQLi', 'XSS'], fill_value=0)
    for name in ['Benign', 'SQLi', 'XSS']:
        lines3.append(f"  pred={name}: {pred_counts[name]}")
    print("\n" + "\n".join(lines3))


if __name__ == '__main__':
    main()
