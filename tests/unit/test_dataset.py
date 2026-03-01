"""Tests for core/feedback/dataset.py"""

import json
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from math_core.momentum import l2_normalize
from core.feedback.dataset import TrajectoryDataset
from core.feedback.models import TurnFeedback, HitMissResult
from core.nerve.models import IntentSignal, PrefetchStrategy
from core.nerve.state import ConversationStateManager


def rand_unit(d=64):
    return l2_normalize(np.random.randn(d).astype(np.float32))


def make_feedback(hits=1, misses=1) -> TurnFeedback:
    results = []
    for i in range(hits):
        results.append(HitMissResult(
            memory_id=f"hit-mem-{i}",
            strategy=PrefetchStrategy.SEMANTIC,
            intent=IntentSignal.DEBUG,
            confidence_at_fetch=0.8,
            string_overlap_score=0.3,
            semantic_sim_score=0.85,
            prevented_retrieval=False,
            is_hit=True,
            hit_signal="semantic",
        ))
    for i in range(misses):
        results.append(HitMissResult(
            memory_id=f"miss-mem-{i}",
            strategy=PrefetchStrategy.GRAPH,
            intent=IntentSignal.DEBUG,
            confidence_at_fetch=0.5,
            string_overlap_score=0.0,
            semantic_sim_score=0.2,
            prevented_retrieval=False,
            is_hit=False,
            hit_signal="miss",
        ))
    fb = TurnFeedback(
        turn_id="t1", session_id="s1",
        turn_index=1, intent=IntentSignal.DEBUG,
        results=results,
    )
    fb.compute_aggregates()
    return fb


def make_state() -> ConversationStateManager:
    state = ConversationStateManager(dim=64)
    for _ in range(3):
        state.push(rand_unit(), 0.85)
    return state


def make_dataset():
    """Dataset with mocked MetaStore."""
    meta = AsyncMock()
    meta._db = AsyncMock()
    meta._db.execute = AsyncMock()
    meta._db.commit = AsyncMock()
    meta._db.executescript = AsyncMock()

    class MockCursor:
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): pass
        def __await__(self):
            async def _coro(): return self
            return _coro().__await__()
        async def fetchone(self): return {"n": 5, "sessions": 2}
        async def fetchall(self): return []
        
    cursor_mock = MockCursor()
    meta._db.execute = MagicMock(return_value=cursor_mock)

    return TrajectoryDataset(meta), meta


@pytest.mark.asyncio
async def test_record_builds_sample():
    dataset, _ = make_dataset()
    feedback = make_feedback(hits=2, misses=1)
    state    = make_state()
    sample   = await dataset.record(feedback, state, switch_score=0.1)

    assert sample.session_id == "s1"
    assert sample.turn_index == 1
    assert sample.intent == IntentSignal.DEBUG.value
    assert len(sample.hit_memory_ids) == 2
    assert len(sample.miss_memory_ids) == 1

@pytest.mark.asyncio
async def test_record_C_t_is_list():
    dataset, _ = make_dataset()
    sample = await dataset.record(make_feedback(), make_state())
    assert isinstance(sample.C_t, list)
    assert len(sample.C_t) == 64

@pytest.mark.asyncio
async def test_record_hit_strategies():
    dataset, _ = make_dataset()
    sample = await dataset.record(make_feedback(hits=1, misses=0), make_state())
    assert PrefetchStrategy.SEMANTIC.value in sample.hit_strategies

@pytest.mark.asyncio
async def test_record_persist_called():
    dataset, meta = make_dataset()
    await dataset.record(make_feedback(), make_state())
    meta._db.execute.assert_called()
    meta._db.commit.assert_called()

@pytest.mark.asyncio
async def test_persist_failure_does_not_raise():
    dataset, meta = make_dataset()
    meta._db.execute.side_effect = Exception("DB error")
    # Should not raise — errors are swallowed in persist
    await dataset.record(make_feedback(), make_state())

@pytest.mark.asyncio
async def test_get_stats_returns_dict():
    dataset, _ = make_dataset()
    stats = await dataset.get_stats()
    assert "total_samples" in stats
    assert "total_sessions" in stats

