"""Semantic edge definitions and utilities."""

semantic_pairs = [
    ('script', 'src'), ('script', 'eval'), ('onerror', 'eval'),
    ('img', 'onerror'), ('svg', 'onload'), ('select', 'from'),
    ('union', 'select'), ('order', 'by'), ('group', 'by')
]


def semantic_edges(tokens, window=15):
    """Return list of directed edge pairs (i,j) for semantic matches within window."""
    edges = []
    n = len(tokens)
    for i in range(n):
        for j in range(i + 1, min(i + window, n)):
            if (tokens[i], tokens[j]) in semantic_pairs:
                edges.append((i, j))
                edges.append((j, i))
    return edges

