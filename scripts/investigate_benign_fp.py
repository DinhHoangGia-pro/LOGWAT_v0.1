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
RESULTS_DIR = ROOT / 'results'
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
    assert len(df) == 90, f"expected 90 rows, got {len(df)}"

    model = HeavyWebGNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    graphs = []
    for _, row in df.iterrows():
        g = build_graph(str(row['content']), int(row['attack_type']), str(row['source_uid']))
        graphs.append(g)

    y_pred = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=8, shuffle=False):
            logits = model(batch.x, batch.edge_index, batch.batch)
            y_pred.extend(int(p) for p in logits.argmax(dim=1).tolist())

    df['pred'] = y_pred
    df['pred_name'] = df['pred'].map(LABELS)
    df['num_nodes'] = [g.num_nodes for g in graphs]
    df['num_edges'] = [g.num_edges for g in graphs]

    lines = []

    # ---- Task 1: raw samples for benign header_field and json_field ----
    lines.append("=" * 90)
    lines.append("TASK 1: Noi dung that + nhan du doan cho benign_header_field va benign_json_field")
    lines.append("=" * 90)
    for technique in ['header_field', 'json_field']:
        subset = df[(df['class_name'] == 'Benign') & (df['technique'] == technique)]
        lines.append(f"\n--- technique={technique} (n={len(subset)}) ---")
        for _, row in subset.iterrows():
            lines.append(
                f"[{row['source_uid']}] true=Benign pred={row['pred_name']} "
                f"nodes={row['num_nodes']} edges={row['num_edges']} content={row['content']!r}"
            )

    print("\n".join(lines))

    with open(DATA_DIR / 'benign_fp_samples.txt', 'w') as f:
        f.write("\n".join(lines) + "\n")

    # ---- Task 2: prediction distribution over all 90 samples ----
    lines2 = []
    lines2.append("=" * 90)
    lines2.append("TASK 2: Phan bo du doan tren toan bo 90 mau held-out (that = 30/30/30)")
    lines2.append("=" * 90)
    true_counts = df['class_name'].value_counts().reindex(['Benign', 'SQLi', 'XSS'], fill_value=0)
    pred_counts = df['pred_name'].value_counts().reindex(['Benign', 'SQLi', 'XSS'], fill_value=0)
    lines2.append(f"\nPhan bo THAT (label):")
    for name in ['Benign', 'SQLi', 'XSS']:
        lines2.append(f"  {name} (label={ {v:k for k,v in LABELS.items()}[name] }): {true_counts[name]}")
    lines2.append(f"\nPhan bo DU DOAN (GATv2):")
    for name in ['Benign', 'SQLi', 'XSS']:
        lines2.append(f"  {name} (pred={ {v:k for k,v in LABELS.items()}[name] }): {pred_counts[name]}")
    lines2.append(f"\nTong so mau: {len(df)}")
    lines2.append(f"So mau du doan dung: {(df['class_name'] == df['pred_name']).sum()} / {len(df)}")

    print("\n" + "\n".join(lines2))

    with open(DATA_DIR / 'prediction_distribution.txt', 'w') as f:
        f.write("\n".join(lines2) + "\n")

    # ---- Task 3: graph size (num_nodes, num_edges) for the 2 failing benign cells ----
    lines3 = []
    lines3.append("=" * 90)
    lines3.append("TASK 3: Kich thuoc graph (num_nodes/num_edges) cua 2 cell benign bi sai,")
    lines3.append("so sanh voi mean da biet: Benign=41.2, SQLi=10.7, XSS=21.7")
    lines3.append("=" * 90)
    for technique in ['header_field', 'json_field']:
        subset = df[(df['class_name'] == 'Benign') & (df['technique'] == technique)]
        lines3.append(f"\n--- technique={technique} (n={len(subset)}) ---")
        for _, row in subset.iterrows():
            lines3.append(
                f"[{row['source_uid']}] num_nodes={row['num_nodes']} num_edges={row['num_edges']} "
                f"pred={row['pred_name']}"
            )
        mean_nodes = subset['num_nodes'].mean()
        mean_edges = subset['num_edges'].mean()
        lines3.append(f"  -> mean num_nodes={mean_nodes:.1f}, mean num_edges={mean_edges:.1f}")
        lines3.append(f"     so voi known means: Benign=41.2, SQLi=10.7, XSS=21.7")
        closest = min(KNOWN_MEANS.items(), key=lambda kv: abs(kv[1] - mean_nodes))
        lines3.append(f"     gan nhat voi lop: {closest[0]} (mean={closest[1]})")

    print("\n" + "\n".join(lines3))

    with open(DATA_DIR / 'benign_fp_graph_sizes.txt', 'w') as f:
        f.write("\n".join(lines3) + "\n")

    print("\nDa ghi: data/benign_fp_samples.txt, data/prediction_distribution.txt, data/benign_fp_graph_sizes.txt")


if __name__ == '__main__':
    main()
