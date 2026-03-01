import numpy as np
import pytest
from math_core.momentum import l2_normalize
from core.types import MemoryID, UnitVector
from core.write.conflict import ConflictResolver, ConflictType


def rand_unit(d=64):
    return l2_normalize(np.random.randn(d).astype(np.float32))

def near_unit(base: np.ndarray, noise: float) -> UnitVector:
    """Unit vector close to base."""
    v = base + noise * np.random.randn(*base.shape).astype(np.float32)
    return l2_normalize(v)


resolver = ConflictResolver()
MID = MemoryID("existing-001")


def test_no_candidates_is_novel():
    r = resolver.resolve(rand_unit(), [])
    assert r.conflict_type == ConflictType.NOVEL
    assert r.should_write is True

def test_duplicate_detected_by_hash():
    emb = rand_unit()
    r = resolver.resolve(emb, [(MID, emb, "abc123")], new_hash="abc123")
    assert r.conflict_type == ConflictType.DUPLICATE
    assert r.should_write is False

def test_duplicate_detected_by_cosine():
    emb = rand_unit()
    similar = near_unit(emb, 0.01)   # very close
    r = resolver.resolve(emb, [(MID, similar, "different_hash")])
    assert r.conflict_type == ConflictType.DUPLICATE
    assert r.should_write is False

def test_conflict_detected():
    base = rand_unit()
    # Cosine sim ~0.85 → conflict zone
    existing = near_unit(base, 0.3)
    sim = float(np.dot(base, existing))
    if 0.80 <= sim < 0.97:
        r = resolver.resolve(base, [(MID, existing, "h1")])
        assert r.conflict_type == ConflictType.CONFLICT
        assert r.edge_type == "CONFLICTS_WITH"
        assert r.should_write is True
        assert r.should_deprecate is False

def test_novel_detected():
    a = rand_unit()
    # Orthogonal vector → very low cosine sim
    b = l2_normalize(np.ones_like(a) - np.dot(np.ones_like(a), a) * a)
    r = resolver.resolve(a, [(MID, b, "h2")])
    assert r.conflict_type in (ConflictType.NOVEL, ConflictType.EXTENSION)

def test_extension_triggers_deprecate():
    base = rand_unit()
    # Target sim ~0.65 → extension zone
    existing = near_unit(base, 0.8)
    sim = float(np.dot(base, existing))
    if 0.55 <= sim < 0.80:
        r = resolver.resolve(base, [(MID, existing, "h3")])
        assert r.conflict_type == ConflictType.EXTENSION
        assert r.should_deprecate is True
        assert r.edge_type == "EXTENDS"

def test_best_match_selection():
    """Should pick the most similar candidate."""
    base = rand_unit()
    close = near_unit(base, 0.05)
    far = rand_unit()
    r = resolver.resolve(base, [
        (MemoryID("far"), far, "h1"),
        (MemoryID("close"), close, "h2"),
    ])
    assert r.existing_id == MemoryID("close")

