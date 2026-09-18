import argparse
import base64
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader

from src.bag.graph_builder import build_single_graph
from src.models.logwat import HeavyWebGNN
from src.preprocessing.normalization import classify_request

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
HELDOUT_PATH = DATA_DIR / 'heldout_keyword_test.csv'
RESULT_PATH = RESULTS_DIR / 'heldout_matrix_full.csv'
MODEL_PATH = DATA_DIR / 'models_pretrained' / 'best_web_gnn_seed42.pth'

LABELS = {0: 'Benign', 1: 'SQLi', 2: 'XSS'}
TECHNIQUES = {
    0: ['field_query', 'header_field', 'json_field'],
    1: ['sql_new_commands', 'comment_splitting', 'case_mixing'],
    2: ['data_uri_base64', 'event_handler_focus', 'svg_script_variant'],
}


def b64_payload(text: str) -> str:
    return base64.b64encode(text.encode('utf-8')).decode('ascii')


def make_matrix_rows():
    rows = []
    cell_specs = {
        (0, 'field_query'): [
            f"GET /home?page={i}&mode=dashboard HTTP/1.1" for i in range(10)
        ],
        (0, 'header_field'): [
            f"Authorization: Bearer token_{i}; X-Trace: session-{i}" for i in range(10)
        ],
        (0, 'json_field'): [
            f"{{\"page\":\"dashboard\",\"user\":\"alice{i}\",\"locale\":\"en-US\"}}" for i in range(10)
        ],
        (1, 'sql_new_commands'): [
            f"login={i};TRUNCATE TABLE audit_log_{i};--" for i in range(10)
        ],
        (1, 'comment_splitting'): [
            f"p={i};UNI/**/ON ALL SEL/**/ECT * FR/**/OM users WH/**/ERE id={i};--" for i in range(10)
        ],
        (1, 'case_mixing'): [
            f"q={i};uN/**/ioN aLl sEl/**/eCt * frOm admins WH/**/ere id={i};--" for i in range(10)
        ],
        (2, 'data_uri_base64'): [
            "data:text/html;base64," + b64_payload(f"<svg xmlns='http://www.w3.org/2000/svg' data-id={i}></svg>")
            for i in range(10)
        ],
        (2, 'event_handler_focus'): [
            f"<svg xmlns='http://www.w3.org/2000/svg' onfocus=payload_{i}(1)>x</svg>" for i in range(10)
        ],
        (2, 'svg_script_variant'): [
            "<svg><a href='data:text/html;base64," + b64_payload(f"<script>trigger({i})</script>") + "'>x</a></svg>"
            for i in range(10)
        ],
    }

    for label, techniques in TECHNIQUES.items():
        for technique in techniques:
            for i, content in enumerate(cell_specs[(label, technique)]):
                rows.append({
                    'source_uid': f'matrix_{LABELS[label].lower()}_{technique}_{i:02d}',
                    'content': content,
                    'attack_type': label,
                    'source': f'matrix_{LABELS[label].lower()}_{technique}',
                    'class_name': LABELS[label],
                    'technique': technique,
                })

    df = pd.DataFrame(rows)

    forbidden = ['union', 'select', 'information_schema', '<script', 'onerror=', 'onload=', 'onclick=', 'javascript:', 'alert(', 'eval(', 'document.cookie', 'window.location']
    for _, row in df.iterrows():
        c = str(row['content']).lower()
        if row['class_name'] in ('SQLi', 'XSS'):
            if any(key in c for key in forbidden):
                raise RuntimeError(f"Forbidden literal still appears in {row['source_uid']}: {row['content']}")
    return df


def evaluate_group(df: pd.DataFrame, model_path=MODEL_PATH,
                    use_seq=True, use_skip=True, use_sem=True, use_edge_attr=False):
    model = HeavyWebGNN(use_edge_attr=use_edge_attr)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    output_rows = []
    for label in [0, 1, 2]:
        for technique in TECHNIQUES[label]:
            subset = df[(df['class_name'] == LABELS[label]) & (df['technique'] == technique)].copy()
            if subset.empty:
                continue

            y_true = subset['attack_type'].astype(int).tolist()
            graphs = []
            for _, row in subset.iterrows():
                graphs.append(build_single_graph(
                    str(row['content']), source_uid=str(row['source_uid']), attack_type=int(row['attack_type']),
                    use_seq=use_seq, use_skip=use_skip, use_sem=use_sem, use_edge_attr=use_edge_attr,
                ))

            y_pred_gat = []
            y_pred_rule = []
            with torch.no_grad():
                for batch in DataLoader(graphs, batch_size=8, shuffle=False):
                    ea = batch.edge_attr if use_edge_attr else None
                    logits = model(batch.x, batch.edge_index, batch.batch, edge_attr=ea)
                    y_pred_gat.extend(int(p) for p in logits.argmax(dim=1).tolist())

            for _, row in subset.iterrows():
                pred_rule = int(classify_request(str(row['content']), None) if classify_request(str(row['content']), None) in (0, 1, 2) else 0)
                y_pred_rule.append(pred_rule)

            output_rows.append({
                'label': label,
                'class_name': LABELS[label],
                'technique': technique,
                'n_samples': len(subset),
                'gatv2_accuracy': float(accuracy_score(y_true, y_pred_gat)),
                'string_matching_accuracy': float(accuracy_score(y_true, y_pred_rule)),
                'gatv2_correct': int(sum(1 for a, b in zip(y_true, y_pred_gat) if a == b)),
                'string_matching_correct': int(sum(1 for a, b in zip(y_true, y_pred_rule) if a == b)),
            })

    return pd.DataFrame(output_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=str(MODEL_PATH))
    parser.add_argument('--result_path', default=str(RESULT_PATH))
    parser.add_argument('--no_seq', action='store_true')
    parser.add_argument('--no_skip', action='store_true')
    parser.add_argument('--no_sem', action='store_true')
    parser.add_argument('--use_edge_attr', action='store_true')
    args = parser.parse_args()

    df = make_matrix_rows()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(HELDOUT_PATH, index=False)

    results = evaluate_group(
        df, model_path=args.model_path,
        use_seq=not args.no_seq, use_skip=not args.no_skip, use_sem=not args.no_sem,
        use_edge_attr=args.use_edge_attr,
    )
    result_path = Path(args.result_path)
    results.to_csv(result_path, index=False)
    print(f"rows_written={len(df)}")
    print(f"saved_heldout={HELDOUT_PATH}")
    print(f"saved_matrix={result_path}")
    print(results.to_string(index=False))


if __name__ == '__main__':
    main()
