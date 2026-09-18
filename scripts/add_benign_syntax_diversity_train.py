"""Add TRAIN-only Benign rows with header-style and JSON-style syntax.

Verified before writing this script (not assumed): of the 1689 train
Benign rows with num_nodes<=15, 0/1689 contain ':' and 0/1689 contain
'{'/'}' -- 1685/1689 (99.8%) are 'key=value&key=value' query-string style.
So it isn't that benign train examples are "too long" relative to the
held-out benign_header_field/benign_json_field cells; it's that train
Benign has essentially zero syntactic coverage of header- or JSON-shaped
content at all, regardless of length. This script adds that coverage.

Field/header names are deliberately DISJOINT from the ones the held-out
matrix uses (build_heldout_matrix_eval.py: 'Authorization'/'X-Trace' for
header_field, 'page'/'user'/'locale' for json_field), so the held-out
cells stay a genuine test of syntactic generalization rather than
something the model was directly trained on.

Append-only: never rewrites existing rows (same discipline as the two
previous train-augmentation scripts in this session).
"""
import csv
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
AUGMENTED_CSV = DATA_DIR / 'augmented_web_attack.csv'
REPORT_PATH = DATA_DIR / 'benign_syntax_diversity_train_report.txt'

SEED = 42
N_HEADER_STYLE = 2500
N_JSON_STYLE = 2500

# Header names/values disjoint from held-out's Authorization/X-Trace.
HEADER_POOL = [
    ('Content-Type', ['application/json', 'application/x-www-form-urlencoded',
                       'text/html; charset=utf-8', 'multipart/form-data']),
    ('Accept-Encoding', ['gzip, deflate', 'gzip, deflate, br', 'identity']),
    ('Cache-Control', ['no-cache', 'no-cache; max-age=0', 'max-age=3600', 'no-store']),
    ('Accept-Language', ['en-US,en;q=0.9', 'vi-VN,vi;q=0.8', 'fr-FR,fr;q=0.7']),
    ('Connection', ['keep-alive', 'close']),
    ('Accept', ['*/*', 'text/html,application/xhtml+xml', 'application/json']),
    ('Vary', ['Accept-Encoding', 'Origin']),
    ('Server', ['nginx/1.18.0', 'Apache/2.4.41', 'gunicorn/20.1.0']),
]
HEADER_DYNAMIC = [
    ('X-Request-ID', lambda rng: f'req-{rng.randrange(16**8):08x}'),
    ('X-Correlation-ID', lambda rng: f'corr-{rng.randrange(16**10):010x}'),
    ('X-Client-Version', lambda rng: f'{rng.randint(1,9)}.{rng.randint(0,20)}.{rng.randint(0,99)}'),
    ('X-Request-Time', lambda rng: str(rng.randint(1_000_000_000, 1_999_999_999))),
    ('ETag', lambda rng: f'"{rng.randrange(16**16):016x}"'),
]

# JSON field names disjoint from held-out's page/user/locale.
JSON_FIELDS = {
    'status': ['ok', 'pending', 'error', 'success', 'failed'],
    'action': ['login', 'logout', 'update', 'create', 'delete', 'refresh'],
    'type': ['event', 'request', 'log', 'metric'],
    'method': ['GET', 'POST', 'PUT', 'DELETE'],
    'path': ['/api/v1/items', '/api/v1/orders', '/health', '/metrics'],
    'level': ['info', 'warn', 'error', 'debug'],
    'region': ['us-east-1', 'eu-west-1', 'ap-southeast-1'],
}
JSON_FIELDS_DYNAMIC = {
    'timestamp': lambda rng: str(rng.randint(1_600_000_000, 1_999_999_999)),
    'count': lambda rng: str(rng.randint(0, 500)),
    'id': lambda rng: str(rng.randint(1000, 999999)),
    'code': lambda rng: str(rng.randint(100, 599)),
    'retries': lambda rng: str(rng.randint(0, 5)),
    'success': lambda rng: rng.choice(['true', 'false']),
}


def make_header_content(rng):
    n_pairs = rng.randint(1, 3)
    static_pairs = [(name, rng.choice(values)) for name, values in HEADER_POOL]
    dynamic_pairs = [(name, fn(rng)) for name, fn in HEADER_DYNAMIC]
    pool = static_pairs + dynamic_pairs
    chosen = rng.sample(pool, k=min(n_pairs, len(pool)))
    return '; '.join(f'{name}: {value}' for name, value in chosen)


def make_json_content(rng):
    n_pairs = rng.randint(2, 4)
    static_keys = [(k, rng.choice(v)) for k, v in JSON_FIELDS.items()]
    dynamic_keys = [(k, fn(rng)) for k, fn in JSON_FIELDS_DYNAMIC.items()]
    pool = static_keys + dynamic_keys
    chosen = rng.sample(pool, k=min(n_pairs, len(pool)))
    body = ','.join(f'"{k}":"{v}"' if not v.replace('.', '', 1).isdigit() and v not in ('true', 'false')
                     else f'"{k}":{v}' for k, v in chosen)
    return '{' + body + '}'


def main():
    rng = random.Random(SEED)

    df = pd.read_csv(AUGMENTED_CSV)
    n_before = len(df)

    if df['source'].eq('benign_short_synthetic').any():
        raise RuntimeError(
            "Dữ liệu nguồn 'benign_short_synthetic' đã tồn tại trong augmented_web_attack.csv "
            "— dừng để tránh append trùng. Xoá thủ công nếu thực sự muốn chạy lại."
        )

    benign_frames = df[df['attack_type'] == 0].reset_index(drop=True)
    if len(benign_frames) == 0:
        raise RuntimeError('No benign frames available as row templates')

    new_rows = []
    seq = 0

    for _ in range(N_HEADER_STYLE):
        content = make_header_content(rng)
        template = benign_frames.iloc[seq % len(benign_frames)].to_dict()
        template['content'] = content
        template['attack_type'] = 0
        template['source'] = 'benign_short_synthetic'
        template['source_uid'] = f'benign_short_synthetic_header_{seq}'
        new_rows.append(template)
        seq += 1

    for _ in range(N_JSON_STYLE):
        content = make_json_content(rng)
        template = benign_frames.iloc[seq % len(benign_frames)].to_dict()
        template['content'] = content
        template['attack_type'] = 0
        template['source'] = 'benign_short_synthetic'
        template['source_uid'] = f'benign_short_synthetic_json_{seq}'
        new_rows.append(template)
        seq += 1

    new_df = pd.DataFrame(new_rows).reindex(columns=df.columns)

    # Append-only: never re-serialize existing rows (a prior read+rewrite
    # round trip this session double-escaped already-escaped backslash
    # content in unrelated rows).
    with open(AUGMENTED_CSV, 'a', newline='') as f:
        new_df.to_csv(f, index=False, header=False, quoting=csv.QUOTE_ALL, escapechar='\\')

    n_after = n_before + len(new_df)

    lines = []
    lines.append("=== Train-only Benign syntax diversity: header-style + json-style ===")
    lines.append(f"Verified before generation: of 1689 train Benign rows with num_nodes<=15,")
    lines.append(f"0/1689 contained ':' and 0/1689 contained '{{'/'}}'; 1685/1689 were key=value&key=value.")
    lines.append(f"Header/field names deliberately disjoint from held-out's Authorization/X-Trace and page/user/locale.")
    lines.append("")
    lines.append(f"seed={SEED}")
    lines.append(f"header-style rows added: {N_HEADER_STYLE}")
    lines.append(f"json-style rows added:   {N_JSON_STYLE}")
    lines.append(f"dataset rows before: {n_before}")
    lines.append(f"dataset rows after:  {n_after}")
    lines.append("")
    lines.append("=== Sample header-style (first 10) ===")
    for r in new_rows[:10]:
        lines.append(f"  {r['source_uid']}: {r['content']}")
    lines.append("")
    lines.append("=== Sample json-style (first 10) ===")
    for r in new_rows[N_HEADER_STYLE:N_HEADER_STYLE + 10]:
        lines.append(f"  {r['source_uid']}: {r['content']}")
    report = "\n".join(lines)
    REPORT_PATH.write_text(report + "\n")
    print(report)


if __name__ == '__main__':
    main()
