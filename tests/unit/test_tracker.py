"""Tests for core/feedback/tracker.py"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
import numpy as np

from math_core.momentum import l2_normalize
from core.feedback.tracker import FeedbackTracker
from core.feedback.detector import HitMissDetector
from core.nerve.models import IntentSignal
from core.staging.cache import StagingCache
from core.staging.models import StagedChunk, StagedMemory
from core.nerve.models import Prediction, PrefetchStrategy


def rand_unit(d=64):
    return l2_normalize(np.random.randn(d).astype(np.float32))


def make_staged_memory(content="some content", conf=0.8):
    pred = Prediction(
        query_vector=rand_unit(), query_text="test",
        graph_seeds=[], annotation_tags=[],
        confidence=conf, strategy=PrefetchStrategy.SEMANTIC,
        intent=IntentSignal.DEBUG, k_steps=1,
    )
    chunk = StagedChunk(
        chunk_id="c1", parent_id="m1", chunk_index=0,
        content=content, tokens=20, score=conf, source_type="prose",
    )
    sm = StagedMemory(
        prediction=pred, chunks=[chunk],
        raw_confidence=conf, rerank_score=1.0,
    )
    sm.was_injected = True
    return sm


def make_tracker(staged_memories=None):
    cache = AsyncMock(spec=StagingCache)
    cache.drain_for_feedback = AsyncMock(
        return_value=staged_memories or []
    )
    cache.mark_used = AsyncMock()

    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=rand_unit())

    detector = HitMissDetector()
    tracker = FeedbackTracker(
        cache=cache, detector=detector,
        embedder=embedder, session_id="test-session",
    )
    return tracker, cache, embedder


@pytest.mark.asyncio
async def test_evaluate_turn_no_staged():
    tracker, _, _ = make_tracker(staged_memories=[])
    feedback = await tracker.evaluate_turn("response text", IntentSignal.EXPLORE)
    assert feedback.total_staged == 0
    assert feedback.results == []

@pytest.mark.asyncio
async def test_evaluate_turn_with_staged():
    sm = make_staged_memory(content="verify token auth session login here")
    tracker, cache, embedder = make_tracker([sm])
    feedback = await tracker.evaluate_turn(
        "the verify token auth session login here was reviewed",
        IntentSignal.DEBUG,
    )
    assert feedback.total_staged == 1
    assert len(feedback.results) == 1

@pytest.mark.asyncio
async def test_evaluate_increments_turn_index():
    tracker, _, _ = make_tracker([])
    await tracker.evaluate_turn("r1", IntentSignal.EXPLORE)
    await tracker.evaluate_turn("r2", IntentSignal.DEBUG)
    assert tracker._turn_index == 2

@pytest.mark.asyncio
async def test_evaluate_calls_embedder():
    sm = make_staged_memory()
    tracker, _, embedder = make_tracker([sm])
    await tracker.evaluate_turn("response", IntentSignal.EXPLORE)
    embedder.embed.assert_called_once()

@pytest.mark.asyncio
async def test_feedback_aggregates_computed():
    sm = make_staged_memory(content="verify token auth session login here")
    tracker, _, _ = make_tracker([sm])
    feedback = await tracker.evaluate_turn(
        "the verify token auth session login here checked",
        IntentSignal.DEBUG,
    )
    assert feedback.hit_rate >= 0.0
    assert feedback.total_staged == feedback.total_hits + feedback.total_misses

@pytest.mark.asyncio
async def test_mark_used_called_on_hits():
    sm = make_staged_memory(content="verify token auth session login here")
    tracker, cache, _ = make_tracker([sm])
    feedback = await tracker.evaluate_turn(
        "verify token auth session login here was checked",
        IntentSignal.DEBUG,
    )
    if feedback.total_hits > 0:
        cache.mark_used.assert_called()

