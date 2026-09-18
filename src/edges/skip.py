"""Skip edges."""

def skip_edges(nodes, k=2):
    """Bidirectional edges between tokens k apart (i <-> i+k)."""
    n = len(nodes)
    edges = []
    for i in range(n):
        if i < n - k:
            edges.append((i, i + k))
            edges.append((i + k, i))
    return edges
