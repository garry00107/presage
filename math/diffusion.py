"""
Personalized PageRank (PPR) over the memory graph.

Given a seed memory node, diffuses relevance through the graph to surface
structurally connected memories (callers, dependencies, related files).

Equation:
    r^(0) = e_seed                      (one-hot on seed)
    r^(t+1) = α·W·r^(t) + (1-α)·e_seed (power iteration)

Where W is the column-stochastic adjacency matrix (row-normalized adj).
Converges in ~10 iterations for α=0.85.

This is equivalent to heat diffusion on the graph, biased to teleport
back to the seed — ensuring seed relevance is maintained.
"""


def personalized_pagerank(
    adj: dict[str, list[str]],
    seed_id: str,
    alpha: float = 0.85,
    iters: int = 10,
    min_score: float = 0.01,
) -> dict[str, float]:
    """
    Sparse PPR via power iteration.

    Args:
        adj:      {node_id: [neighbor_id, ...]} adjacency list (directed)
        seed_id:  starting node
        alpha:    damping factor (0.85 standard)
        iters:    power iteration steps (10 is sufficient for convergence)
        min_score: filter out nodes below this relevance score

    Returns:
        {node_id: relevance_score}, excluding seed and low-score nodes.
    """
    if seed_id not in adj:
        return {}

    nodes = list(adj.keys())
    scores: dict[str, float] = {n: 0.0 for n in nodes}
    scores[seed_id] = 1.0

    for _ in range(iters):
        new_scores: dict[str, float] = {n: 0.0 for n in nodes}
        for node, neighbors in adj.items():
            if not neighbors:
                continue
            share = alpha * scores[node] / len(neighbors)
            for nb in neighbors:
                if nb in new_scores:
                    new_scores[nb] += share
        # Teleport back to seed
        new_scores[seed_id] += 1.0 - alpha
        scores = new_scores

    return {
        nid: s for nid, s in scores.items()
        if nid != seed_id and s >= min_score
    }

