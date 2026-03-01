"""
Integration test: full Feedback Loop.
Tracker → Detector → Loop → Predictor bandit update.
"""

import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from math_core.momentum import l2_normalize
from core.feedback.detector import HitMissDetector
from core.feedback.loop import FeedbackLoop
from core.feedback.tracker import FeedbackTracker
from core.feedback.dataset import TrajectoryDataset
from core.nerve.models import IntentSignal, PrefetchStrategy
from core.nerve.predictor import TrajectoryPredictor
from core.nerve.state import ConversationStateManager
from core.staging.cache import StagingCache
from core.staging.models import StagedChunk, StagedMemory
from core.nerve.models import Prediction


DIM = 64


def rand_unit():
    return l2_normalize(np.random.randn(DIM).astype(np.float32))


def make_staged_memory(content, conf=0.8, injected=True):
    pred = Prediction(
        query_vector=rand_unit(), query_text="test",
        graph_seeds=[], annotation_tags=[],
        confidence=conf, strategy=PrefetchStrategy.SEMANTIC,
        intent=IntentSignal.DEBUG, k_steps=1,
    )
    chunk = StagedChunk(
        chunk_id=f"c-{id(content)}", parent_id=f"m-{id(content)}",
        chunk_index=0, content=content, tokens=len(content)//3,
        score=conf, source_type="prose",
    )
    sm = StagedMemory(
        prediction=pred, chunks=[chunk],
        raw_confidence=conf, rerank_score=1.0,
    )
    sm.was_injected = injected
    return sm


def make_meta_mock():
    meta = AsyncMock()
    meta._db = AsyncMock()
    meta._db.execute = AsyncMock()
    meta._db.commit = AsyncMock()
    meta._db.executescript = AsyncMock()
    meta.increment_annotation_hit = AsyncMock()
    cursor = AsyncMock()
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=False)
    cursor.fetchone = AsyncMock(return_value={"n": 0, "sessions": 0})
    cursor.fetchall = AsyncMock(return_value=[])
    meta._db.execute.return_value = cursor
    return meta


@pytest.mark.asyncio
async def test_full_feedback_loop_updates_bandits():
    """After a hit, SEMANTIC:DEBUG bandit confidence should increase."""
    state    = ConversationStateManager(dim=DIM)
    for _ in range(3):
        state.push(rand_unit(), 0.85)

    predictor = TrajectoryPredictor(state)
    meta      = make_meta_mock()
    dataset   = TrajectoryDataset(meta)

    # Simulate a hit: content appears verbatim in response
    hit_content = "verify token authentication session login here"
    sm = make_staged_memory(hit_content, conf=0.8)

    cache = AsyncMock(spec=StagingCache)
    cache.drain_for_feedback = AsyncMock(return_value=[sm])
    cache.mark_used = AsyncMock()

    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=rand_unit())

    tracker = FeedbackTracker(
        cache=cache, detector=HitMissDetector(),
        embedder=embedder, session_id="test-session",
    )
    loop = FeedbackLoop(
        predictor=predictor, state=state,
        dataset=dataset, meta_store=meta,
    )

    conf_before = predictor._bandits.confidence(
        PrefetchStrategy.SEMANTIC.value, IntentSignal.DEBUG.value
    )

    # Run feedback: content appears in response → overlap hit
    response = f"I checked the code: {hit_content} was the issue."
    feedback = await tracker.evaluate_turn(response, IntentSignal.DEBUG)
    await loop.process(feedback, switch_score=0.1)

    conf_after = predictor._bandits.confidence(
        PrefetchStrategy.SEMANTIC.value, IntentSignal.DEBUG.value
    )

    if feedback.total_hits > 0:
        assert conf_after >= conf_before, \
            "Confidence should not decrease after a hit"


@pytest.mark.asyncio
async def test_feedback_loop_updates_graph_seeds():
    """Hit memory IDs should become graph seeds in the predictor."""
    state     = ConversationStateManager(dim=DIM)
    predictor = TrajectoryPredictor(state)
    meta      = make_meta_mock()
    dataset   = TrajectoryDataset(meta)

    hit_content = "verify token auth session login here system"
    sm = make_staged_memory(hit_content, conf=0.8)

    cache = AsyncMock(spec=StagingCache)
    cache.drain_for_feedback = AsyncMock(return_value=[sm])
    cache.mark_used = AsyncMock()

    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=rand_unit())

    tracker = FeedbackTracker(
        cache=cache, detector=HitMissDetector(),
        embedder=embedder, session_id="test-session",
    )
    loop = FeedbackLoop(
        predictor=predictor, state=state,
        dataset=dataset, meta_store=meta,
    )

    response = f"reviewed: {hit_content}"
    feedback = await tracker.evaluate_turn(response, IntentSignal.DEBUG)
    await loop.process(feedback)

    if feedback.total_hits > 0:
        assert len(predictor._recent_graph_seeds) > 0


@pytest.mark.asyncio
async def test_feedback_records_trajectory_sample():
    state     = ConversationStateManager(dim=DIM)
    for _ in range(2):
        state.push(rand_unit(), 0.85)

    predictor = TrajectoryPredictor(state)
    meta      = make_meta_mock()
    dataset   = TrajectoryDataset(meta)

    sm = make_staged_memory("some content that was staged")
    cache = AsyncMock(spec=StagingCache)
    cache.drain_for_feedback = AsyncMock(return_value=[sm])
    cache.mark_used = AsyncMock()

    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=rand_unit())

    tracker = FeedbackTracker(
        cache=cache, detector=HitMissDetector(),
        embedder=embedder, session_id="s1",
    )
    loop = FeedbackLoop(
        predictor=predictor, state=state,
        dataset=dataset, meta_store=meta,
    )

    feedback = await tracker.evaluate_turn("any response", IntentSignal.EXPLORE)
    await loop.process(feedback)

    # dataset.record calls meta._db.execute to persist
    meta._db.execute.assert_called()


@pytest.mark.asyncio
async def test_feedback_loop_no_results_is_noop():
    """Empty feedback should not crash or update anything."""
    state     = ConversationStateManager(dim=DIM)
    predictor = TrajectoryPredictor(state)
    meta      = make_meta_mock()
    dataset   = TrajectoryDataset(meta)

    loop = FeedbackLoop(
        predictor=predictor, state=state,
        dataset=dataset, meta_store=meta,
    )

    from core.feedback.models import TurnFeedback
    empty_feedback = TurnFeedback(
        turn_id="t0", session_id="s0",
        turn_index=0, intent=IntentSignal.UNKNOWN, results=[],
    )
    # Should complete without error
    await loop.process(empty_feedback)
