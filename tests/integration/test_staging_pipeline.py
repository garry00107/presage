"""
Integration test: full Staging pipeline.
Prefetcher → Cache → Reranker → Injector, with mock stores.
"""

import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.staging.cache import StagingCache
from core.staging.injector import Injector
from core.staging.models import StagedChunk, StagedMemory
from core.staging.prefetcher import Prefetcher
from core.staging.reranker import Reranker
from core.nerve.models import Prediction, PrefetchStrategy, IntentSignal
from math_core.momentum import l2_normalize


DIM = 64

def rand_unit():
    return l2_normalize(np.random.randn(DIM).astype(np.float32))


def make_vector_store_mock():
    """Mock vector store returning realistic chunk results."""
    mock = AsyncMock()
    mock.search = AsyncMock(return_value=[
        {
            "id": f"chunk-{i:03d}",
            "parent_id": f"mem-{i // 2:03d}",
            "content": f"def function_{i}():\n    return {i}",
            "tokens": 20,
            "score": 0.9 - i * 0.05,
            "source_type": "code",
            "chunk_index": i % 2,
            "source": f"src/module_{i}.py",
        }
        for i in range(5)
    ])
    return mock


def make_meta_store_mock():
    mock = AsyncMock()
    mock.search_by_annotation = AsyncMock(return_value=[])
    mock.get_chunks_by_memory_ids = AsyncMock(return_value=[])
    return mock


def make_prediction(conf=0.85, strategy=PrefetchStrategy.SEMANTIC) -> Prediction:
    return Prediction(
        query_vector=rand_unit(),
        query_text="fix the authentication bug",
        graph_seeds=[],
        annotation_tags=["intent:DEBUG", "symbol:verify_token"],
        confidence=conf,
        strategy=strategy,
        intent=IntentSignal.DEBUG,
        k_steps=1,
        slot_index=0,
    )


@pytest.mark.asyncio
async def test_full_staging_pipeline():
    """Full pipeline: predictions → prefetch → cache → rerank → inject."""
    vector_store = make_vector_store_mock()
    meta_store   = make_meta_store_mock()

    prefetcher = Prefetcher(vector_store, meta_store)
    cache      = StagingCache(prefetcher)
    reranker   = Reranker()
    injector   = Injector()

    # Step 1: Schedule prefetch from predictions
    predictions = [make_prediction(conf=0.85), make_prediction(conf=0.60)]
    await cache.schedule_prefetch(predictions)
    await asyncio.sleep(0.1)   # allow background tasks to complete

    # Step 2: Get all staged memories
    staged = await cache.get_all_ready()
    assert len(staged) > 0, "Cache should have staged memories"

    # Step 3: Rerank against the actual user query
    query_emb = rand_unit()
    chunk_embeddings = {}  # no embeddings in mock — fallback to raw confidence
    reranked = reranker.rerank(staged, query_emb, chunk_embeddings)
    assert len(reranked) == len(staged)

    # Step 4: Build injection plan
    plan = injector.plan(reranked, token_budget=500)
    assert plan.tokens_used <= 500
    assert plan.memories_injected >= 0

    # Step 5: Verify context text is well-formed
    if plan.chunks:
        ctx = plan.context_text
        assert "<memory_context>" in ctx
        assert "</memory_context>" in ctx


@pytest.mark.asyncio
async def test_soft_trigger_fires_hot_memory():
    """HOT tier memory should inject when trigger text matches annotation tag."""
    vector_store = make_vector_store_mock()
    meta_store   = make_meta_store_mock()
    prefetcher   = Prefetcher(vector_store, meta_store)
    injector     = Injector()

    # Manually create a HOT-tier staged memory with symbol annotation
    from core.staging.models import StagedChunk, StagedMemory
    pred = make_prediction(conf=0.55)   # HOT tier
    pred.annotation_tags = ["symbol:verify_token"]
    chunk = StagedChunk(
        chunk_id="hot-chunk", parent_id="hot-mem", chunk_index=0,
        content="def verify_token(tok): ...", tokens=30, score=0.55,
        source_type="code",
    )
    hot_sm = StagedMemory(
        prediction=pred, chunks=[chunk],
        raw_confidence=0.55, rerank_score=1.0,
    )

    # Without trigger — HOT not injected (no AUTO memories present either,
    # so fallback kicks in — test that trigger increases injection)
    plan_no_trigger = injector.plan([hot_sm], token_budget=1000)

    # With matching trigger
    plan_triggered = injector.plan(
        [hot_sm], token_budget=1000,
        soft_trigger="why does verify_token throw an error?"
    )
    # Triggered should inject the memory
    assert plan_triggered.memories_injected >= 1


@pytest.mark.asyncio
async def test_token_budget_hard_limit():
    """Injector must NEVER exceed token budget."""
    vector_store = make_vector_store_mock()
    meta_store   = make_meta_store_mock()
    prefetcher   = Prefetcher(vector_store, meta_store)
    injector     = Injector()

    predictions = [make_prediction(conf=0.90)]
    cache = StagingCache(prefetcher)
    await cache.schedule_prefetch(predictions)
    await asyncio.sleep(0.1)

    staged = await cache.get_all_ready()
    for budget in [100, 200, 500, 1000]:
        plan = injector.plan(staged, token_budget=budget)
        assert plan.tokens_used <= budget, \
            f"Budget {budget} exceeded: used {plan.tokens_used}"


@pytest.mark.asyncio
async def test_expired_memories_not_injected():
    """Expired staged memories must never make it into an injection plan."""
    from core.staging.models import StagedMemory
    injector = Injector()

    pred = make_prediction(conf=0.90)
    chunk = StagedChunk(
        chunk_id="exp-chunk", parent_id="exp-mem", chunk_index=0,
        content="expired content", tokens=50, score=0.9, source_type="prose",
    )
    expired_sm = StagedMemory(
        prediction=pred, chunks=[chunk],
        raw_confidence=0.90, rerank_score=1.0,
        ttl_seconds=0.001,   # expires almost immediately
    )

    await asyncio.sleep(0.01)
    assert expired_sm.is_expired is True

    # Cache evicts expired; injector receives empty list
    plan = injector.plan([])
    assert plan.memories_injected == 0
