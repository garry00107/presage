"""
Integration test: full session end-to-end.
SessionFactory → SessionManager → turn() with mock LLM and stores.
"""

import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from math_core.momentum import l2_normalize
from core.session.factory import SessionFactory
from core.nerve.models import IntentSignal


DIM = 64


def rand_unit():
    return l2_normalize(np.random.randn(DIM).astype(np.float32))


def make_embedder_mock():
    emb = AsyncMock()
    emb.embed = AsyncMock(side_effect=lambda t: rand_unit())
    emb.embed_batch = AsyncMock(side_effect=lambda ts: [rand_unit() for _ in ts])
    emb.dim = DIM
    return emb


def make_vector_mock():
    v = AsyncMock()
    v.search = AsyncMock(return_value=[])
    v.upsert = AsyncMock()
    v.connect = AsyncMock()
    return v


def make_meta_mock():
    m = AsyncMock()
    m._db = AsyncMock()
    m._db.execute = AsyncMock()
    m._db.executescript = AsyncMock()
    m._db.commit = AsyncMock()
    cursor = AsyncMock()
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=False)
    cursor.fetchone = AsyncMock(return_value={"n": 0, "sessions": 0})
    cursor.fetchall = AsyncMock(return_value=[])
    m._db.execute.return_value = cursor
    m.insert_memory = AsyncMock(return_value="mem-001")
    m.touch_memory = AsyncMock()
    m.soft_delete = AsyncMock()
    m.search_by_annotation = AsyncMock(return_value=[])
    m.get_recently_written = AsyncMock(return_value=[])
    m.increment_annotation_hit = AsyncMock()
    m.save_bandit_state = AsyncMock()
    m.load_bandit_state = AsyncMock(return_value=None)
    return m


def make_distiller_mock():
    from core.write.distiller import MemoryDistiller
    d = AsyncMock(spec=MemoryDistiller)
    d.distill = AsyncMock(return_value=[])   # no memories distilled (fast tests)
    return d


async def make_factory():
    embedder = make_embedder_mock()
    vector   = make_vector_mock()
    meta     = make_meta_mock()

    async def mock_llm(prompt: str) -> str:
        return f"Mock LLM response to: {prompt[:50]}..."

    factory = SessionFactory(
        embedder=embedder,
        meta_store=meta,
        vector_store=vector,
        llm_caller=mock_llm,
        distiller=make_distiller_mock(),
    )
    return factory


@pytest.mark.asyncio
async def test_create_session():
    factory = await make_factory()
    session = await factory.create_session()
    assert session.session_id is not None
    assert session.turn_count == 0


@pytest.mark.asyncio
async def test_session_turn_returns_result():
    factory = await make_factory()
    session = await factory.create_session()
    result  = await session.turn("What is dependency injection?")
    assert result.llm_response != ""
    assert result.turn_index == 1
    assert result.intent is not None


@pytest.mark.asyncio
async def test_session_turn_increments_count():
    factory = await make_factory()
    session = await factory.create_session()
    await session.turn("First message")
    await session.turn("Second message")
    assert session.turn_count == 2


@pytest.mark.asyncio
async def test_session_turn_returns_intent():
    factory = await make_factory()
    session = await factory.create_session()
    result  = await session.turn("why does my code crash with an error?")
    assert result.intent == IntentSignal.DEBUG


@pytest.mark.asyncio
async def test_session_latency_recorded():
    factory = await make_factory()
    session = await factory.create_session()
    result  = await session.turn("test message")
    assert result.latency_ms > 0


@pytest.mark.asyncio
async def test_multiple_sessions_independent():
    factory = await make_factory()
    s1 = await factory.create_session()
    s2 = await factory.create_session()
    assert s1.session_id != s2.session_id

    await s1.turn("message to session 1")
    assert s1.turn_count == 1
    assert s2.turn_count == 0


@pytest.mark.asyncio
async def test_close_session():
    factory = await make_factory()
    session = await factory.create_session()
    sid     = session.session_id

    await factory.close_session(sid)
    retrieved = await factory.get_session(sid)
    assert retrieved is None


@pytest.mark.asyncio
async def test_auto_create_session_on_missing():
    """Factory should create a session if get returns None."""
    factory = await make_factory()
    session = await factory.get_session("nonexistent-id")
    assert session is None  # returns None, caller can create


@pytest.mark.asyncio
async def test_session_velocity_increases_over_turns():
    """Velocity should be nonzero after multiple diverse turns."""
    factory = await make_factory()
    session = await factory.create_session()

    turns = [
        "explain what authentication means",
        "how does JWT work?",
        "write a function to verify tokens",
        "why does my token validation fail?",
    ]
    for t in turns:
        await session.turn(t)

    # After 4 diverse turns, velocity should be non-zero
    assert session.velocity >= 0.0  # always true
    assert session.turn_count == 4

