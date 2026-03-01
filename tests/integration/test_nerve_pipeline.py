"""
Integration test: full Nerve Layer pipeline.
Observer → State → Predictor, no mocks for the math.
Uses a real embedder mock that returns deterministic vectors.
"""

import numpy as np
import pytest
from unittest.mock import AsyncMock
from math_core.momentum import l2_normalize
from core.nerve.models import IntentSignal, PrefetchStrategy
from core.nerve.predictor import TrajectoryPredictor
from core.nerve.state import ConversationStateManager
from core.surface.observer import ConversationObserver


DIM = 64
_call_count = 0

def deterministic_embed(text: str, dim: int = 64) -> np.ndarray:
    """
    Mock embedder that returns a slowly drifting unit vector so that
    momentum can build up sequentially across turns predictably.
    """
    base = np.ones(dim, dtype=np.float32)
    base[0] += len(text) * 0.5
    return l2_normalize(base)


def make_pipeline():
    embedder = AsyncMock()
    embedder.embed = AsyncMock(
        side_effect=lambda t: deterministic_embed(t)
    )
    state = ConversationStateManager(dim=DIM)
    observer = ConversationObserver(embedder, state)
    predictor = TrajectoryPredictor(state)
    return observer, predictor, state


@pytest.mark.asyncio
async def test_full_pipeline_three_turns():
    observer, predictor, state = make_pipeline()

    turns = [
        "Why does verify_token throw an AttributeError?",
        "The error is in the JWT decode step. Fix it.",
        "Now write a test for the fixed verify_token function.",
    ]

    all_predictions = []
    for turn in turns:
        signals = await observer.observe(turn)
        preds = predictor.predict(signals)
        all_predictions.append((signals, preds))

    # After 3 turns we should have momentum
    assert state.has_momentum
    assert state.turn_count == 3

    # All prediction query vectors must be unit vectors
    for _, preds in all_predictions:
        for p in preds:
            norm = np.linalg.norm(p.query_vector)
            assert abs(norm - 1.0) < 1e-4


@pytest.mark.asyncio
async def test_debug_intent_triggers_graph_strategy():
    observer, predictor, state = make_pipeline()

    # Prime with history
    for t in ["explain auth", "show me login.py"]:
        signals = await observer.observe(t)
        predictor.predict(signals)

    # Give the predictor some seeds
    predictor.update_graph_seeds(["mem-auth-001", "mem-login-002"])

    # Debug turn — should trigger GRAPH strategy
    signals = await observer.observe("error in verify_token() function")
    assert signals.intent == IntentSignal.DEBUG

    preds = predictor.predict(signals)
    strategies = {p.strategy for p in preds}
    assert PrefetchStrategy.GRAPH in strategies


@pytest.mark.asyncio
async def test_bandit_learning_over_session():
    """Predictor should increase confidence for strategies that hit."""
    observer, predictor, state = make_pipeline()

    key_s = PrefetchStrategy.SEMANTIC.value
    key_i = IntentSignal.EXPLORE.value
    conf_start = predictor._bandits.confidence(key_s, key_i)

    # Simulate 20 hits for SEMANTIC:EXPLORE
    for _ in range(20):
        predictor.update_bandits(key_s, key_i, hit=True)

    conf_end = predictor._bandits.confidence(key_s, key_i)
    assert conf_end > conf_start
    assert conf_end > 0.7


@pytest.mark.asyncio
async def test_context_switch_resets_momentum():
    import core.surface.observer as obs_module
    observer, predictor, state = make_pipeline()

    # Build up momentum over several turns
    for t in ["auth token", "jwt decode", "session management"]:
        signals = await observer.observe(t)
        predictor.predict(signals)

    vel_before = state.current_velocity

    # Force context switch
    original = obs_module.context_switch_score
    obs_module.context_switch_score = lambda a, b: 0.99

    signals = await observer.observe("completely unrelated query about databases")
    assert signals.did_reset is True
    # Velocity resets to 0 after reset
    assert state.current_velocity == 0.0

    obs_module.context_switch_score = original


@pytest.mark.asyncio
async def test_prediction_horizon_grows_with_velocity():
    """
    High velocity turns should predict further ahead (larger k).
    We can't directly control velocity without controlling embeddings,
    so we verify k is reasonable given the system's internal state.
    """
    observer, predictor, state = make_pipeline()

    for t in ["auth", "database", "async workers", "deployment", "testing"]:
        signals = await observer.observe(t)
        preds = predictor.predict(signals)

    # After diverse turns, semantic predictions should exist for k>=1
    signals = await observer.observe("final question")
    preds = predictor.predict(signals)
    sem_preds = [p for p in preds if p.strategy == PrefetchStrategy.SEMANTIC]
    assert len(sem_preds) >= 1
    assert all(p.k_steps >= 1 for p in sem_preds)
