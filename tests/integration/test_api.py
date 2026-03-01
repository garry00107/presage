"""
Integration tests for the FastAPI REST API.
Uses TestClient — no real stores, all mocked.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from math_core.momentum import l2_normalize
from api.server import app
from api.deps import set_factory


DIM = 64


def rand_unit():
    return l2_normalize(np.random.randn(DIM).astype(np.float32))


def make_mock_factory():
    """Build a minimal mock SessionFactory for API tests."""
    from core.session.factory import SessionFactory

    factory = AsyncMock(spec=SessionFactory)
    factory._sessions = {}
    factory._embedder = AsyncMock()
    factory._embedder.dim = DIM
    factory._embedder.embed = AsyncMock(return_value=rand_unit())
    factory._vector = AsyncMock()
    factory._vector.search = AsyncMock(return_value=[])
    factory._meta = AsyncMock()
    factory._meta._db = AsyncMock()
    factory._meta.insert_memory = AsyncMock(return_value="mem-001")
    factory._dataset = AsyncMock()
    factory._dataset.get_stats = AsyncMock(return_value={
        "total_samples": 0, "total_sessions": 0, "by_intent": {}
    })
    factory._chunker = MagicMock()
    factory._chunker.chunk = MagicMock(return_value=[])
    factory._annotator = MagicMock()
    factory._annotator.annotate = MagicMock(return_value=[])

    # Mock create_session
    async def mock_create(session_id=None, restore_bandits=True):
        from unittest.mock import MagicMock
        session = AsyncMock()
        session.session_id = session_id or "test-session-id"
        session.turn_count = 0
        session._cache = AsyncMock()
        session._cache.slot_summary = MagicMock(return_value=[
            {"slot": i, "status": "empty"} for i in range(10)
        ])

        async def mock_turn(user_message, source="", stream=False):
            from core.session.manager import TurnResult
            from core.staging.models import InjectionPlan
            from core.nerve.models import IntentSignal
            plan = InjectionPlan(
                chunks=[], tokens_used=0, tokens_budget=4096,
                memories_injected=0, staged_memories=[],
            )
            return TurnResult(
                session_id=session.session_id,
                turn_index=1,
                user_message=user_message,
                llm_response="Mock response",
                injection_plan=plan,
                intent=IntentSignal.EXPLORE,
                velocity=0.12,
                hit_rate=0.5,
                tokens_injected=0,
                memories_injected=0,
                latency_ms=42.0,
            )

        session.turn = mock_turn
        factory._sessions[session.session_id] = session
        return session

    factory.create_session = mock_create
    factory.get_session = AsyncMock(
        side_effect=lambda sid: factory._sessions.get(sid)
    )
    factory.close_session = AsyncMock(
        side_effect=lambda sid: factory._sessions.pop(sid, None)
    )
    factory.close_all = AsyncMock()

    return factory


@pytest.fixture
def client():
    factory = make_mock_factory()
    set_factory(factory)
    with TestClient(app) as c:
        yield c, factory


def test_health_check(client):
    c, factory = client
    factory._meta._db.execute = AsyncMock()
    resp = c.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data


def test_create_session(client):
    c, _ = client
    resp = c.post("/v1/session", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "created_at" in data


def test_create_session_with_id(client):
    c, _ = client
    resp = c.post("/v1/session", json={"session_id": "my-custom-id"})
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "my-custom-id"


def test_submit_turn(client):
    c, _ = client
    # Create session first
    create_resp = c.post("/v1/session", json={})
    sid = create_resp.json()["session_id"]

    # Submit turn
    resp = c.post("/v1/turn", json={
        "session_id": sid,
        "message": "What is authentication?",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert "intent" in data
    assert "latency_ms" in data


def test_turn_response_has_slots(client):
    c, _ = client
    create_resp = c.post("/v1/session", json={})
    sid = create_resp.json()["session_id"]
    resp = c.post("/v1/turn", json={"session_id": sid, "message": "test"})
    data = resp.json()
    assert "staged_slots" in data


def test_close_session(client):
    c, _ = client
    create_resp = c.post("/v1/session", json={})
    sid = create_resp.json()["session_id"]
    resp = c.delete(f"/v1/session/{sid}")
    assert resp.status_code == 200


def test_stats_endpoint(client):
    c, _ = client
    resp = c.get("/v1/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "active_sessions" in data
    assert "trajectory_samples" in data


def test_search_memories(client):
    c, _ = client
    resp = c.get("/v1/memory/search", params={"query": "authentication", "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert data["query"] == "authentication"
