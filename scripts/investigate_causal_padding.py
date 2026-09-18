from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.bag.node_features import get_node_features
from src.edges.semantic import semantic_edges
from src.models.logwat import HeavyWebGNN
from src.preprocessing.tokenizer import web_security_tokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
MODEL_PATH = DATA_DIR / 'models_pretrained' / 'best_web_gnn_seed42.pth'

LABELS = {0: 'Benign', 1: 'SQLi', 2: 'XSS'}


def build_graph(content: str, source_uid: str) -> Data:
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
    return Data(x=x, edge_index=edge_index, y=torch.tensor([0], dtype=torch.long), source_uid=source_uid)


def predict(model, content: str, source_uid: str):
    g = build_graph(content, source_uid)
    with torch.no_grad():
        for batch in DataLoader([g], batch_size=1, shuffle=False):
            logits = model(batch.x, batch.edge_index, batch.batch)
            pred = int(logits.argmax(dim=1).item())
    return pred, g.num_nodes, g.num_edges


def main():
    model = HeavyWebGNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    lines = []
    lines.append("=" * 90)
    lines.append("CAUSAL PROBE: sua kich thuoc graph (padding/truncation), giu nguyen loai noi dung,")
    lines.append("xem prediction co doi theo kich thuoc graph hay khong")
    lines.append("=" * 90)

    # ---------------------------------------------------------------
    # Case 1: benign header_field (known: 12 nodes, pred=SQLi) -> PAD
    # ---------------------------------------------------------------
    before_content = 'Authorization: Bearer token_0; X-Trace: session-0'
    after_content = before_content + '; ' + '; '.join(['X-Custom-Header: value'] * 3)

    pred_before, nodes_before, edges_before = predict(model, before_content, 'case1_before')
    pred_after, nodes_after, edges_after = predict(model, after_content, 'case1_after')

    lines.append("\n--- CASE 1: benign header_field -> PADDED (them filler vo hai) ---")
    lines.append(f"TRUOC: content={before_content!r}")
    lines.append(f"       num_nodes={nodes_before} num_edges={edges_before} prediction={LABELS[pred_before]}")
    lines.append(f"SAU:   content={after_content!r}")
    lines.append(f"       num_nodes={nodes_after} num_edges={edges_after} prediction={LABELS[pred_after]}")
    lines.append(f"=> Doi tu {LABELS[pred_before]} sang {LABELS[pred_after]} khi num_nodes tang {nodes_before} -> {nodes_after}"
                  if pred_before != pred_after else
                  f"=> KHONG doi, van la {LABELS[pred_after]} du num_nodes tang {nodes_before} -> {nodes_after}")

    # ---------------------------------------------------------------
    # Case 2: benign field_query (known: 16 nodes, pred=Benign, dung) -> TRUNCATE
    # ---------------------------------------------------------------
    before_content2 = 'GET /home?page=0&mode=dashboard HTTP/1.1'
    after_content2 = 'GET /home?page=0&mode=dashboard'

    pred_before2, nodes_before2, edges_before2 = predict(model, before_content2, 'case2_before')
    pred_after2, nodes_after2, edges_after2 = predict(model, after_content2, 'case2_after')

    lines.append("\n--- CASE 2: benign field_query (dang dung) -> TRUNCATED (cat bot) ---")
    lines.append(f"TRUOC: content={before_content2!r}")
    lines.append(f"       num_nodes={nodes_before2} num_edges={edges_before2} prediction={LABELS[pred_before2]}")
    lines.append(f"SAU:   content={after_content2!r}")
    lines.append(f"       num_nodes={nodes_after2} num_edges={edges_after2} prediction={LABELS[pred_after2]}")
    lines.append(f"=> Doi tu {LABELS[pred_before2]} sang {LABELS[pred_after2]} khi num_nodes giam {nodes_before2} -> {nodes_after2}"
                  if pred_before2 != pred_after2 else
                  f"=> KHONG doi, van la {LABELS[pred_after2]} du num_nodes giam {nodes_before2} -> {nodes_after2}")

    # ---------------------------------------------------------------
    # Case 3: SQLi comment_splitting (known: 37 nodes, pred=Benign) -> RUT GON /**/
    # ---------------------------------------------------------------
    before_content3 = 'p=0;UNI/**/ON ALL SEL/**/ECT * FR/**/OM users WH/**/ERE id=0;--'
    after_content3 = 'p=0;UNION ALL SELECT * FROM users--'

    pred_before3, nodes_before3, edges_before3 = predict(model, before_content3, 'case3_before')
    pred_after3, nodes_after3, edges_after3 = predict(model, after_content3, 'case3_after')

    lines.append("\n--- CASE 3: SQLi comment_splitting -> RUT GON (bo /**/,  giu payload UNION ALL SELECT) ---")
    lines.append(f"TRUOC: content={before_content3!r}")
    lines.append(f"       num_nodes={nodes_before3} num_edges={edges_before3} prediction={LABELS[pred_before3]}")
    lines.append(f"SAU:   content={after_content3!r}")
    lines.append(f"       num_nodes={nodes_after3} num_edges={edges_after3} prediction={LABELS[pred_after3]}")
    lines.append(f"=> Doi tu {LABELS[pred_before3]} sang {LABELS[pred_after3]} khi num_nodes giam {nodes_before3} -> {nodes_after3}"
                  if pred_before3 != pred_after3 else
                  f"=> KHONG doi, van la {LABELS[pred_after3]} du num_nodes giam {nodes_before3} -> {nodes_after3}")

    lines.append("\n" + "=" * 90)
    lines.append("TOM TAT")
    lines.append("=" * 90)
    lines.append(f"Case 1 (benign header_field, pad {nodes_before}->{nodes_after} nodes): "
                  f"{LABELS[pred_before]} -> {LABELS[pred_after]}")
    lines.append(f"Case 2 (benign field_query, truncate {nodes_before2}->{nodes_after2} nodes): "
                  f"{LABELS[pred_before2]} -> {LABELS[pred_after2]}")
    lines.append(f"Case 3 (SQLi comment_splitting, shrink {nodes_before3}->{nodes_after3} nodes): "
                  f"{LABELS[pred_before3]} -> {LABELS[pred_after3]}")

    output = "\n".join(lines)
    print(output)

    with open(DATA_DIR / 'causal_padding_test.txt', 'w') as f:
        f.write(output + "\n")

    print("\nDa ghi: data/causal_padding_test.txt")


if __name__ == '__main__':
    main()
