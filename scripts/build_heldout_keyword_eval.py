import csv
import pickle
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.bag.node_features import get_node_features
from src.edges.semantic import semantic_edges
from src.models.logwat import HeavyWebGNN
from src.preprocessing.normalization import classify_request
from src.preprocessing.tokenizer import web_security_tokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
MODEL_PATH = DATA_DIR / 'models_pretrained' / 'best_web_gnn_seed42.pth'
HELDOUT_CSV = DATA_DIR / 'heldout_keyword_test.csv'
RESULT_CSV = RESULTS_DIR / 'heldout_keyword_comparison.csv'


def build_graph_from_content(content: str, attack_type: int, source_uid: str) -> Data:
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


def make_heldout_rows():
    pool_df = pd.read_csv(DATA_DIR / 'sqli_payload_pool.csv') if (DATA_DIR / 'sqli_payload_pool.csv').exists() else pd.DataFrame(columns=['payload'])
    pool_set = set(pool_df['payload'].dropna().astype(str).str.lower().tolist())

    sqli_payloads = [
        "login=1;TRUNCATE TABLE audit_log;--",
        "user=admin';BEGIN;TRUNCATE TABLE sessions;COMMIT;--",
        "ID=2;UNI/**/ON ALL SEL/**/ECT user,pwd FROM accounts WHERE id=1;--",
        "q=1;/*x*/UNI/**/ON /*z*/SEL/**/ECT * FROM users WHERE role='admin';--",
        "payload=1;DELETE FROM sessions WHERE session_id='abc';--",
        "p=0; /*ctx*/TRUNCATE TABLE app_log; /*nd*/",
        "v=7;UPDATE users SET role='admin' WHERE id=8;--",
        "u=go; /* stealth */ DELETE FROM tokens WHERE owner='root';--",
    ]
    xss_payloads = [
        'data:text/html;base64,PHNjcmlwdD5wcmludCgxKTwvc2NyaXB0Pg==',
        'data:text/html;base64,PHNjcmlwdD5wcm9taXQoKTs8L3NjcmlwdD4=',
        'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxzY3JpcHQ+cHJpbnQoMSk8L3NjcmlwdD48L3N2Zz4=',
        'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxzY3JpcHQ+ZG9jdW1lbnQuY29va2llPC9zY3JpcHQ+PC9zdmc+',
        'data:text/html;base64,PHNjcmlwdD5zdHlsZS5mbG9hdD0xPC9zY3JpcHQ+',
        'data:text/html;base64,PHNjcmlwdD53aW5kb3cubG9jYXRpb24gPSAiaHR0cDovL2V4YW1wbGUuY29tIjwvc2NyaXB0Pg==',
        'data:application/xhtml+xml;base64,PHNjcmlwdD5jb25zb2xlLmxvZygxKTwvc2NyaXB0Pg==',
        'data:text/html;base64,PHNjcmlwdD5pbm5lckhUTUw9IjEiPC9zY3JpcHQ+',
    ]

    benign_payloads = [
        "GET /home?page=dashboard HTTP/1.1",
        "username=alice&password=correcthorsebattery",
        "search=hello world and welcome",
        "product_id=42&category=books",
        "header: Accept-Language en-US",
        "content-type: application/json",
    ]

    def unique_new(items):
        out = []
        seen = set()
        for item in items:
            k = item.lower()
            if k in seen or k in pool_set:
                continue
            seen.add(k)
            out.append(item)
        return out

    rows = []
    for idx, payload in enumerate(unique_new(sqli_payloads)):
        rows.append({
            'source_uid': f'heldout_sqli_{idx}',
            'content': payload,
            'attack_type': 1,
            'source': 'heldout_keyword_sqli',
        })
    for idx, payload in enumerate(unique_new(xss_payloads)):
        rows.append({
            'source_uid': f'heldout_xss_{idx}',
            'content': payload,
            'attack_type': 2,
            'source': 'heldout_keyword_xss',
        })
    for idx, payload in enumerate(unique_new(benign_payloads)):
        rows.append({
            'source_uid': f'heldout_benign_{idx}',
            'content': payload,
            'attack_type': 0,
            'source': 'heldout_keyword_benign',
        })

    # keep a strict, reproducible ordering so the output is easy to inspect later
    df = pd.DataFrame(rows, columns=['source_uid', 'content', 'attack_type', 'source'])

    # Final sanity check: these examples must not match the current heuristic Table 1 and must also avoid the pool list.
    bad = []
    forbidden = [
        'union', 'select', 'information_schema', '<script', 'onerror', 'onload',
        'javascript', 'alert(', 'window.location', 'document.cookie', 'onmouseover',
        'eval(', 'srcdoc'
    ]
    for _, r in df.iterrows():
        content = r['content'].lower()
        if content in pool_set:
            bad.append((r['source_uid'], 'in-pool'))
        if any(f in content for f in forbidden):
            bad.append((r['source_uid'], f'forbidden-literal:{next(v for v in forbidden if v in content)}'))
        heuristic = classify_request(r['content'], None)
        if heuristic in (1, 2):
            bad.append((r['source_uid'], f'heuristic-hit:{heuristic}'))
    if bad:
        raise RuntimeError(f'Held-out set still violates constraints: {bad}')

    return df


def evaluate_model_on_heldout(df: pd.DataFrame):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graphs = [build_graph_from_content(row.content, int(row.attack_type), str(row.source_uid)) for _, row in df.iterrows()]
    loader = DataLoader(graphs, batch_size=1, shuffle=False)

    model = HeavyWebGNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            preds = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(batch.y.cpu().tolist())

    return [int(v) for v in y_true], [int(v) for v in y_pred]


def evaluate_string_matcher_on_heldout(df: pd.DataFrame):
    y_true = df['attack_type'].astype(int).tolist()
    y_pred = []
    for _, row in df.iterrows():
        rule = classify_request(row['content'], None)
        pred = rule if rule in (0, 1, 2) else 0
        y_pred.append(int(pred))
    return y_true, y_pred


def main():
    df = make_heldout_rows()
    HELDOUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(HELDOUT_CSV, index=False)

    y_true_model, y_pred_model = evaluate_model_on_heldout(df)
    y_true_rule, y_pred_rule = evaluate_string_matcher_on_heldout(df)

    gat_acc = accuracy_score(y_true_model, y_pred_model)
    rule_acc = accuracy_score(y_true_rule, y_pred_rule)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['GATv2 accuracy', 'string-matching accuracy'])
        writer.writerow([gat_acc, rule_acc])

    print('Held-out rows:', len(df))
    print('GATv2 accuracy:', gat_acc)
    print('String-matching accuracy:', rule_acc)
    print(f'Saved held-out CSV to {HELDOUT_CSV}')
    print(f'Saved comparison CSV to {RESULT_CSV}')


if __name__ == '__main__':
    main()
