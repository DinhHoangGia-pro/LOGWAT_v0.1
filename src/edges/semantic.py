"""Semantic edge definitions and utilities."""

semantic_pairs = [
    ('script', 'src'), ('script', 'eval'), ('onerror', 'eval'),
    ('img', 'onerror'), ('svg', 'onload'), ('select', 'from'),
    ('union', 'select'), ('order', 'by'), ('group', 'by')
]

_KEYWORDS = sorted({word for pair in semantic_pairs for word in pair})

# Punctuation commonly inserted mid-keyword to evade keyword matching
# (e.g. "UNI/**/ON", "SEL--ECT", "sel#ect").
_NOISE_TOKENS = {'/', '*', '-', '#'}


def _reconstruct_keywords(tokens):
    """Merge token fragments split by noise punctuation back into whole words.

    Walks the token list once. Starting at each alphanumeric token, greedily
    extends the merge across runs of noise tokens as long as the growing
    string stays a prefix of some known keyword, so unrelated tokens are
    never fused together. Returns (keyword, anchor_index) pairs, where
    anchor_index is the original token index of the fragment that completed
    the match — used to keep edges anchored to real node indices.
    """
    n = len(tokens)
    matches = []
    i = 0
    while i < n:
        tok = tokens[i]
        if not tok or not tok[0].isalnum():
            i += 1
            continue

        merged = tok
        last_idx = i
        j = i + 1
        while j < n:
            if tokens[j] in _NOISE_TOKENS:
                j += 1
                continue
            if tokens[j] and tokens[j][0].isalnum():
                candidate = merged + tokens[j]
                if any(candidate == kw or kw.startswith(candidate) for kw in _KEYWORDS):
                    merged = candidate
                    last_idx = j
                    j += 1
                    continue
            break

        if merged in _KEYWORDS:
            matches.append((merged, last_idx))
        i = last_idx + 1
    return matches


def semantic_edges(tokens, window=15):
    """Return list of directed edge pairs (i,j) for semantic matches within window.

    Keyword tokens are normalized first (see `_reconstruct_keywords`) so that
    keywords split by noise punctuation (e.g. comment-splitting evasion like
    "UNI/**/ON ... SEL/**/ECT") are still recognized, instead of only matching
    tokens that equal a keyword outright.
    """
    edges = []
    keyword_hits = _reconstruct_keywords(tokens)
    for a in range(len(keyword_hits)):
        word_i, idx_i = keyword_hits[a]
        for b in range(a + 1, len(keyword_hits)):
            word_j, idx_j = keyword_hits[b]
            if idx_j - idx_i >= window:
                break
            if (word_i, word_j) in semantic_pairs:
                edges.append((idx_i, idx_j))
                edges.append((idx_j, idx_i))
    return edges

