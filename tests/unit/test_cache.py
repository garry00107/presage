"""Tests for core/staging/cache.py"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from core.staging.cache import StagingCache
from core.staging.models import StagedChunk, StagedMemory, SlotTier
from core.nerve.models import Prediction, PrefetchStrategy, IntentSignal
from math_core.momentum import l2_normalize
from config.settings import settings


def make_prediction(conf=0.9, strategy=PrefetchStrategy.SEMANTIC) -> Prediction:
    return Prediction(
        query_vector=l2_normalize(np.random.randn(64).astype(np.float32)),
        query_text="test query",
        graph_seeds=[],
        annotation_tags=[],
        confidence=conf,
        strategy=strategy,
        intent=IntentSignal.EXPLORE,
        k_steps=1,
        slot_index=0,
    )


def make_staged_memory(conf=0.9, ttl=120.0) -> StagedMemory:
    pred = make_prediction(conf=conf)
    chunk = StagedChunk(
        chunk_id="chunk-001",
        parent_id="mem-001",
        content="def verify_token(): pass",
        tokens=10,
        score=conf,
        source_type="code",
        chunk_index=0,
    )
    return StagedMemory(
        prediction=pred,
        chunks=[chunk],
        raw_confidence=conf,
        rerank_score=1.0,
        ttl_seconds=ttl,
    )


def make_cache() -> tuple[StagingCache, AsyncMock]:
    prefetcher = AsyncMock()
    prefetcher.fetch = AsyncMock(return_value=make_staged_memory())
    return StagingCache(prefetcher), prefetcher


@pytest.mark.asyncio
async def test_cache_starts_empty():
    cache, _ = make_cache()
    result = await cache.get_all_ready()
    assert result == []

@pytest.mark.asyncio
async def test_schedule_prefetch_fills_slots():
    cache, prefetcher = make_cache()
    preds = [make_prediction(conf=0.9) for _ in range(3)]
    await cache.schedule_prefetch(preds)
    # Wait briefly for background tasks
    await asyncio.sleep(0.05)
    ready = await cache.get_all_ready()
    assert len(ready) > 0

@pytest.mark.asyncio
async def test_auto_inject_only_high_confidence():
    cache, _ = make_cache()
    # Manually fill slots with different confidence levels
    async with cache._lock:
        cache._slots[0] = make_staged_memory(conf=0.90)  # AUTO
        cache._slots[1] = make_staged_memory(conf=0.55)  # HOT
        cache._slots[2] = make_staged_memory(conf=0.25)  # WARM

    auto = await cache.get_auto_inject()
    assert len(auto) == 1
    assert auto[0].raw_confidence >= settings.auto_inject_threshold

@pytest.mark.asyncio
async def test_evict_expired_removes_stale():
    cache, _ = make_cache()
    async with cache._lock:
        cache._slots[0] = make_staged_memory(ttl=0.001)  # expires immediately
        cache._slots[1] = make_staged_memory(ttl=120.0)  # stays

    await asyncio.sleep(0.01)
    evicted = await cache.evict_expired()
    assert evicted == 1

    ready = await cache.get_all_ready()
    assert len(ready) == 1

@pytest.mark.asyncio
async def test_mark_injected():
    cache, _ = make_cache()
    sm = make_staged_memory()
    async with cache._lock:
        cache._slots[0] = sm

    await cache.mark_injected(["mem-001"])
    assert sm.was_injected is True

@pytest.mark.asyncio
async def test_slot_summary_format():
    cache, _ = make_cache()
    summary = cache.slot_summary()
    assert len(summary) == settings.slot_count
    assert all("slot" in s for s in summary)
    assert summary[0]["status"] == "empty"

@pytest.mark.asyncio
async def test_cancel_previous_tasks_on_new_prefetch():
    """New prefetch cancels old in-flight tasks."""
    cache, prefetcher = make_cache()

    # Slow prefetcher
    async def slow_fetch(pred):
        await asyncio.sleep(10)
        return make_staged_memory()

    prefetcher.fetch = slow_fetch

    preds1 = [make_prediction()]
    await cache.schedule_prefetch(preds1)
    task_count_before = len(cache._active_tasks)

    # Immediately schedule new prefetch — should cancel old
    preds2 = [make_prediction()]
    await cache.schedule_prefetch(preds2)

    # Old tasks should be cancelled
    assert len(cache._active_tasks) <= len(preds2)

@pytest.mark.asyncio
async def test_prefetch_failure_does_not_crash():
    """Prefetch errors are swallowed — cache stays operational."""
    cache, prefetcher = make_cache()
    prefetcher.fetch = AsyncMock(side_effect=Exception("DB down"))

    preds = [make_prediction()]
    await cache.schedule_prefetch(preds)
    await asyncio.sleep(0.05)

    # Cache should still work
    ready = await cache.get_all_ready()
    assert ready == []   # nothing staged, but no crash

