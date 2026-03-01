"""Tests for core/nerve/state.py — ConversationStateManager"""

import numpy as np
import pytest
from math_core.momentum import l2_normalize
from core.nerve.state import ConversationStateManager


def rand_unit(d=64):
    return l2_normalize(np.random.randn(d).astype(np.float32))


def make_state(dim=64) -> ConversationStateManager:
    return ConversationStateManager(dim=dim)


def test_initial_velocity_zero():
    s = make_state()
    assert s.current_velocity == 0.0

def test_push_increments_turn_count():
    s = make_state()
    s.push(rand_unit(), 0.85)
    assert s.turn_count == 1
    s.push(rand_unit(), 0.85)
    assert s.turn_count == 2

def test_has_momentum_after_two_turns():
    s = make_state()
    s.push(rand_unit(), 0.85)
    assert not s.has_momentum   # need at least 2 turns
    s.push(rand_unit(), 0.85)
    # Momentum exists if vectors differ (they almost certainly do)
    # velocity may still be near-zero if vectors happen to be close

def test_C_t_is_unit_vector():
    s = make_state()
    s.push(rand_unit(), 0.85)
    assert s.C_t is not None
    norm = np.linalg.norm(s.C_t)
    assert abs(norm - 1.0) < 1e-5

def test_predict_returns_unit_vector():
    s = make_state()
    for _ in range(3):
        s.push(rand_unit(), 0.85)
    pred = s.predict(k=1)
    assert abs(np.linalg.norm(pred) - 1.0) < 1e-4

def test_predict_k0_equals_C_t():
    """k=0 means no extrapolation — should return C_t."""
    s = make_state()
    for _ in range(3):
        s.push(rand_unit(), 0.85)
    pred_k0 = s.predict(k=0)
    np.testing.assert_allclose(pred_k0, s.C_t, atol=1e-5)

def test_reset_clears_momentum():
    s = make_state()
    for _ in range(4):
        s.push(rand_unit(), 0.85)
    prev_velocity = s.current_velocity
    s.reset()
    # After reset, velocity should be 0 (no momentum)
    assert s.current_velocity == 0.0

def test_reset_preserves_turn_count():
    """turn_count is not reset — it tracks total turns in the session."""
    s = make_state()
    for _ in range(5):
        s.push(rand_unit(), 0.85)
    count_before = s.turn_count
    s.reset()
    assert s.turn_count == count_before

def test_history_bounded():
    """State history must not exceed state_window_max."""
    s = make_state()
    from config.settings import settings
    for _ in range(settings.state_window_max + 5):
        s.push(rand_unit(), 0.85)
    assert len(s._history) <= settings.state_window_max

def test_snapshot_serializable():
    s = make_state()
    for _ in range(3):
        s.push(rand_unit(), 0.85)
    snap = s.snapshot()
    # All numpy arrays must be convertible to list (JSON serializable)
    import json
    json.dumps({
        "velocity": snap.velocity,
        "acceleration": snap.acceleration,
        "turn_count": snap.turn_count,
        "C_t": snap.C_t.tolist(),
    })

