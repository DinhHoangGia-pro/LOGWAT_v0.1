"""Node feature extraction for web tokens."""
import re
from src.preprocessing.tokenizer import get_entropy

danger_chars = ["'", '"', ";", "--", "#", "/*", "(", ")", "<", ">", "=", "+", "%", ".", "@"]
sql_kw = ['select', 'union', 'where', 'from', 'insert', 'drop', 'limit', 'exec', 'null']
web_kw = ['script', 'alert', 'onerror', 'eval', 'src', 'href', 'javascript']


def get_node_features(token, index, num_nodes):
    """Return a fixed-length feature vector (list of floats) for a token."""
    t = str(token)
    # Basic statistics
    f_stat = [len(t) / 20.0,
              index / (num_nodes + 1) if num_nodes > 0 else 0.0,
              get_entropy(t) / 8.0,
              1.0 if t.isdigit() else 0.0,
              sum(c.isdigit() for c in t) / (len(t) + 1),
              sum(not c.isalnum() for c in t) / (len(t) + 1),
              1.0 if len(t) > 12 else 0.0,
              1.0 if re.search(r"[0-9a-f]{4,}", t) else 0.0]

    f_char = [1.0 if c in t else 0.0 for c in danger_chars]
    f_sql = [1.0 if t == kw else 0.0 for kw in sql_kw]
    f_web = [1.0 if t == kw else 0.0 for kw in web_kw]

    feat = f_stat + f_char + f_sql + f_web
    # pad to 64 dims
    if len(feat) < 64:
        feat += [0.0] * (64 - len(feat))
    else:
        feat = feat[:64]
    return feat
"""Node feature extraction."""

def extract_features(node):
    return {}
