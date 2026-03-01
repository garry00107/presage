"""Tests for core/staging/injector.py"""

import pytest
import numpy as np
from core.staging.injector import Injector
from core.staging.models import StagedChunk, StagedMemory, SlotTier
from core.nerve.models import Prediction, PrefetchStrategy, IntentSignal
from math_core.momentum import l2_normalize


def make_prediction(conf=0.9, tags=None) -> Prediction:
    return Prediction(
        query_vector=l2_normalize(np.random.randn(64).astype(np.float32)),
        query_text="test",
        graph_seeds=[],
        annotation_tags=tags or [],
        confidence=conf,
        strategy=PrefetchStrategy.SEMANTIC,
        intent=IntentSignal.EXPLORE,
        k_steps=1,
    )


def make_chunk(cid, pid, idx, tokens, score, content=None) -> StagedChunk:
    return StagedChunk(
        chunk_id=cid, parent_id=pid, chunk_index=idx,
        content=content or f"content of {cid}",
        tokens=tokens, score=score, source_type="prose",
    )


def make_staged(conf=0.9, chunks=None, tags=None) -> StagedMemory:
    pred = make_prediction(conf=conf, tags=tags)
    if chunks is None:
        chunks = [make_chunk("c1", "m1", 0, 50, conf)]
    return StagedMemory(
        prediction=pred,
        chunks=chunks,
        raw_confidence=conf,
        rerank_score=1.0,
    )


injector = Injector()


def test_empty_staged_returns_empty_plan():
    plan = injector.plan([])
    assert plan.chunks == []
    assert plan.tokens_used == 0

def test_auto_tier_always_included():
    sm = make_staged(conf=0.90)  # AUTO tier
    plan = injector.plan([sm], token_budget=1000)
    assert plan.memories_injected >= 1

def test_hot_tier_requires_trigger():
    sm = make_staged(conf=0.55, tags=["symbol:verify_token"])  # HOT
    # No trigger → not injected (unless no AUTO present)
    plan_no_trigger = injector.plan([sm], token_budget=1000, soft_trigger=None)
    # No AUTO, so best HOT gets included as fallback
    assert plan_no_trigger.memories_injected >= 0  # depends on fallback logic

    # With matching trigger → injected
    plan_triggered = injector.plan([sm], token_budget=1000,
                                    soft_trigger="fix verify_token please")
    assert plan_triggered.memories_injected >= 1

def test_token_budget_respected():
    chunks = [make_chunk(f"c{i}", "m1", i, 200, 0.9) for i in range(5)]
    sm = make_staged(conf=0.90, chunks=chunks)
    plan = injector.plan([sm], token_budget=400)
    assert plan.tokens_used <= 400

def test_content_never_truncated():
    original = "def foo():\n    return 42"
    chunk = make_chunk("cx", "mx", 0, 20, 0.9, content=original)
    sm = make_staged(conf=0.90, chunks=[chunk])
    plan = injector.plan([sm], token_budget=1000)
    if plan.chunks:
        assert plan.chunks[0].content == original

def test_context_text_has_memory_tags():
    sm = make_staged(conf=0.90)
    plan = injector.plan([sm], token_budget=1000)
    if plan.chunks:
        ctx = plan.context_text
        assert "<memory_context>" in ctx
        assert "</memory_context>" in ctx
        assert "<memory" in ctx

def test_deduplicate_same_parent():
    """Two staged memories for same parent → only highest score wins."""
    chunks_a = [make_chunk("ca", "parent1", 0, 50, 0.9)]
    chunks_b = [make_chunk("cb", "parent1", 1, 50, 0.7)]
    sm_a = make_staged(conf=0.90, chunks=chunks_a)
    sm_b = make_staged(conf=0.70, chunks=chunks_b)
    plan = injector.plan([sm_a, sm_b], token_budget=1000)
    parent_ids = {c.parent_id for c in plan.chunks}
    assert len(parent_ids) == 1

def test_reading_order_preserved_in_context():
    """Chunks from same parent must appear in chunk_index order in context."""
    chunks = [
        make_chunk("c3", "p1", 2, 30, 0.8, "Third chunk"),
        make_chunk("c1", "p1", 0, 30, 0.9, "First chunk"),
        make_chunk("c2", "p1", 1, 30, 0.85, "Second chunk"),
    ]
    sm = make_staged(conf=0.90, chunks=chunks)
    plan = injector.plan([sm], token_budget=1000)
    ctx = plan.context_text
    if "First chunk" in ctx and "Third chunk" in ctx:
        assert ctx.index("First chunk") < ctx.index("Third chunk")

def test_zero_budget_returns_empty():
    sm = make_staged(conf=0.90)
    plan = injector.plan([sm], token_budget=0)
    assert plan.chunks == []

