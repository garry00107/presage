"""
0/1 Knapsack for context budget allocation.

Operates on pre-chunked semantic units (never raw content).
Chunks are split at AST/sentence/structure boundaries at write time,
so the knapsack NEVER truncates content — it only selects whole chunks.

DP formulation:
    maximize   Σ v_i · x_i
    subject to Σ w_i · x_i ≤ B
               x_i ∈ {0, 1}

Solution: bottom-up DP, O(N·B) time, O(N·B) space.
For N≤50 and B≤4096: table size ≤ 800KB, runtime < 1ms.
"""

from core.types import Chunk


def knapsack_01(chunks: list[Chunk], budget: int) -> list[Chunk]:
    """
    Select whole chunks to maximize total score within token budget.

    Restores reading order within each parent memory
    (sorted by chunk_index) so injected context is coherent.

    Args:
        chunks: list of Chunk dicts with 'tokens', 'score', 'id',
                'parent_id', 'chunk_index', 'content'
        budget: max tokens to inject

    Returns:
        Selected chunks in (parent_id, chunk_index) order.
        Content is NEVER modified or truncated.
    """
    n = len(chunks)
    if n == 0 or budget <= 0:
        return []

    # Filter out chunks that can never fit
    feasible = [c for c in chunks if c["tokens"] <= budget]
    if not feasible:
        return []

    n = len(feasible)
    # Bottom-up DP table
    # dp[i][w] = max score using first i items with weight budget w
    dp = [[0.0] * (budget + 1) for _ in range(n + 1)]

    for i, chunk in enumerate(feasible, 1):
        w_i = chunk["tokens"]
        v_i = chunk["score"]
        for w in range(budget + 1):
            dp[i][w] = dp[i - 1][w]                    # skip item i
            if w >= w_i:
                take = dp[i - 1][w - w_i] + v_i
                if take > dp[i][w]:
                    dp[i][w] = take                     # take item i

    # Backtrack to recover selected items
    selected_ids: set[str] = set()
    w = budget
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_ids.add(feasible[i - 1]["id"])
            w -= feasible[i - 1]["tokens"]
            if w <= 0:
                break

    selected = [c for c in feasible if c["id"] in selected_ids]

    # Restore reading order: group by parent, sort by chunk_index
    selected.sort(key=lambda c: (c["parent_id"], c["chunk_index"]))
    return selected

