"""Sequential edge induction."""

def sequential_edges(nodes):
    """Bidirectional edges between adjacent tokens (i <-> i+1)."""
    n = len(nodes)
    edges = []
    for i in range(n):
        if i < n - 1:
            edges.append((i, i + 1))
            edges.append((i + 1, i))
    return edges
