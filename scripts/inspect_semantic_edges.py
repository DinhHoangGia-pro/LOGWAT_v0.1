import sys
sys.path.insert(0, '.')
from src.preprocessing.tokenizer import web_security_tokenizer
from src.edges.semantic import semantic_edges

samples = {
    'comment_split (37 node, pred=Benign)': 'p=0;UNI/**/ON ALL SEL/**/ECT * FR/**/OM users WH/**/ERE id=0;--',
    'clean (12 node, pred=SQLi dung)': 'p=0;UNION ALL SELECT * FROM users--',
    'header_field (12 node, pred=SQLi sai)': 'Authorization: Bearer token_0; X-Trace: session-0',
    'header_padded (36 node, pred=XSS sai)': 'Authorization: Bearer token_0; X-Trace: session-0; X-Custom-Header: value; X-Custom-Header: value; X-Custom-Header: value',
}

for name, content in samples.items():
    tokens = web_security_tokenizer(content)
    sem_edges = semantic_edges(tokens, window=15)
    print(f"--- {name} ---")
    print(f"  n_tokens={len(tokens)}")
    print(f"  tokens (10 dau)={tokens[:10]}")
    print(f"  so canh E_sem = {len(sem_edges)}")
    print(f"  danh sach canh E_sem = {sem_edges}")
    print()
