"""Tests for math/knapsack.py"""

import pytest
from math_core.knapsack import knapsack_01


def make_chunk(id, parent, idx, tokens, score, content=None):
    return {
        "id": id, "parent_id": parent, "chunk_index": idx,
        "tokens": tokens, "score": score,
        "content": content or f"content_{id}",
        "source_type": "prose"
    }


def test_empty_chunks():
    assert knapsack_01([], 100) == []

def test_zero_budget():
    chunks = [make_chunk("a", "p1", 0, 10, 1.0)]
    assert knapsack_01(chunks, 0) == []

def test_all_fit():
    chunks = [make_chunk("a", "p1", 0, 10, 0.9),
              make_chunk("b", "p1", 1, 20, 0.8)]
    result = knapsack_01(chunks, 100)
    assert len(result) == 2

def test_optimal_selection():
    """Should pick high-value item even if lower-scoring fills budget."""
    chunks = [
        make_chunk("a", "p1", 0, 90, 10.0),   # best value/token
        make_chunk("b", "p2", 0, 50, 4.0),
        make_chunk("c", "p2", 1, 50, 4.0),
    ]
    result = knapsack_01(chunks, 100)
    ids = {c["id"] for c in result}
    assert "a" in ids, "High-value item should always be selected"

def test_content_never_truncated():
    """CRITICAL: no chunk content should be modified."""
    original = "def foo():\n    return 42\n"
    chunks = [make_chunk("a", "p1", 0, 20, 0.9, content=original)]
    result = knapsack_01(chunks, 100)
    assert result[0]["content"] == original

def test_reading_order_preserved():
    """Chunks from same parent must be in chunk_index order."""
    chunks = [
        make_chunk("c", "p1", 2, 10, 0.9),
        make_chunk("a", "p1", 0, 10, 0.7),
        make_chunk("b", "p1", 1, 10, 0.8),
    ]
    result = knapsack_01(chunks, 100)
    indices = [c["chunk_index"] for c in result]
    assert indices == sorted(indices)

def test_oversized_chunks_excluded():
    chunks = [make_chunk("big", "p1", 0, 500, 99.0)]
    assert knapsack_01(chunks, 100) == []

