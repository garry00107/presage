"""Tests for core/feedback/detector.py"""

import numpy as np
import pytest
from math_core.momentum import l2_normalize
from core.staging.models import StagedChunk, StagedMemory
from core.nerve.models import Prediction, PrefetchStrategy, IntentSignal
from core.feedback.detector import (
    HitMissDetector,
    OVERLAP_HIT_THRESHOLD,
    SEMANTIC_HIT_THRESHOLD,
)


def rand_unit(d=64):
    return l2_normalize(np.random.randn(d).astype(np.float32))


def make_staged(content="def verify_token(): pass", conf=0.8) -> StagedMemory:
    pred = Prediction(
        query_vector=rand_unit(),
        query_text="test",
        graph_seeds=[], annotation_tags=[],
        confidence=conf,
        strategy=PrefetchStrategy.SEMANTIC,
        intent=IntentSignal.DEBUG,
        k_steps=1,
    )
    chunk = StagedChunk(
        chunk_id="c1", parent_id="m1", chunk_index=0,
        content=content, tokens=20, score=conf, source_type="code",
    )
    return StagedMemory(
        prediction=pred, chunks=[chunk],
        raw_confidence=conf, rerank_score=1.0,
    )


detector = HitMissDetector()


# ── Trigram overlap ────────────────────────────────────────────────────────────

def test_trigram_overlap_identical():
    text = "the quick brown fox jumps over the lazy dog"
    overlap = detector._trigram_overlap(text, text)
    assert overlap == 1.0

def test_trigram_overlap_no_match():
    mem  = "authentication token validation function"
    resp = "completely unrelated text about databases"
    overlap = detector._trigram_overlap(mem, resp)
    assert overlap == 0.0

def test_trigram_overlap_partial():
    mem  = "verify token authentication session login"
    resp = "the verify token authentication was checked but the session expired"
    overlap = detector._trigram_overlap(mem, resp)
    assert 0.0 < overlap < 1.0

def test_trigram_overlap_short_text():
    """Text with < 3 tokens returns 0.0 (no trigrams)."""
    overlap = detector._trigram_overlap("hi", "hi there how are you")
    assert overlap == 0.0


# ── Hit detection ──────────────────────────────────────────────────────────────

def test_hit_via_overlap():
    """Memory content appears in response → hit via overlap."""
    content = "verify token authentication session login system"
    sm = make_staged(content=content)
    response = f"I reviewed the code. The {content} logic looks correct."
    result = detector.detect(sm, response, rand_unit())
    assert result.is_hit is True
    assert result.hit_signal == "overlap"

def test_hit_via_semantic():
    """High cosine similarity → hit via semantic."""
    sm = make_staged()
    mem_emb = rand_unit()
    # Make response embedding identical to memory embedding → sim = 1.0
    result = detector.detect(
        sm,
        response_text="completely different text with no overlap",
        response_embedding=mem_emb,
        memory_embedding=mem_emb,
    )
    assert result.is_hit is True
    assert result.hit_signal == "semantic"

def test_miss_no_overlap_low_sim():
    """No overlap and low similarity → miss."""
    sm = make_staged("def verify_token(): pass")
    mem_emb  = rand_unit()
    resp_emb = l2_normalize(-mem_emb + 0.5 * np.random.randn(64))

    # Force low similarity
    result = detector.detect(
        sm,
        response_text="databases are great for storing relational data",
        response_embedding=resp_emb,
        memory_embedding=mem_emb,
    )
    # With low sim and no overlap, should be a miss
    if result.semantic_sim_score <= SEMANTIC_HIT_THRESHOLD and \
       result.string_overlap_score <= OVERLAP_HIT_THRESHOLD:
        assert result.is_hit is False
        assert result.hit_signal == "miss"

def test_detect_batch():
    """detect_batch returns one result per staged memory."""
    memories = [make_staged(f"content {i}") for i in range(5)]
    results = detector.detect_batch(
        memories, "some response text", rand_unit()
    )
    assert len(results) == 5

def test_result_has_correct_strategy():
    sm = make_staged()
    result = detector.detect(sm, "response", rand_unit())
    assert result.strategy == PrefetchStrategy.SEMANTIC

def test_result_has_correct_intent():
    sm = make_staged()
    result = detector.detect(sm, "response", rand_unit())
    assert result.intent == IntentSignal.DEBUG


# ── Edge cases ─────────────────────────────────────────────────────────────────

def test_empty_memory_content():
    sm = make_staged(content="")
    result = detector.detect(sm, "some response", rand_unit())
    # Should not crash; overlap should be 0
    assert result.string_overlap_score == 0.0

def test_empty_response():
    sm = make_staged("def foo(): pass")
    result = detector.detect(sm, "", rand_unit())
    assert result.string_overlap_score == 0.0

def test_no_memory_embedding_uses_overlap_only():
    sm = make_staged("verify token session authentication login here")
    response = "the verify token session authentication login here was checked"
    result = detector.detect(sm, response, rand_unit(), memory_embedding=None)
    # Semantic sim should be 0 (no embedding)
    assert result.semantic_sim_score == 0.0
    # But overlap should fire
    assert result.is_hit is True

