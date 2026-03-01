"""
Property-based and unit tests for math/momentum.py
Run: pytest tests/unit/test_momentum.py -v
"""

import numpy as np
import pytest
from hypothesis import given, settings as h_settings, strategies as st
from math_core.momentum import (
    l2_normalize, conversation_state,
    momentum_tangent, predict_future_state
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def rand_unit(d: int = 64) -> np.ndarray:
    v = np.random.randn(d).astype(np.float32)
    return l2_normalize(v)


def assert_unit(v: np.ndarray, tol: float = 1e-5):
    norm = np.linalg.norm(v)
    assert abs(norm - 1.0) < tol, f"Expected unit vector, got norm={norm}"


# ── l2_normalize ───────────────────────────────────────────────────────────────

def test_l2_normalize_unit():
    v = np.array([3.0, 4.0])
    u = l2_normalize(v)
    assert_unit(u)

def test_l2_normalize_already_unit():
    v = np.array([1.0, 0.0, 0.0])
    u = l2_normalize(v)
    assert_unit(u)
    np.testing.assert_allclose(u, v, atol=1e-6)

def test_l2_normalize_zero_safe():
    v = np.zeros(4)
    u = l2_normalize(v)
    assert not np.any(np.isnan(u)), "Zero vector should not produce NaN"

@given(st.lists(st.floats(-1e3, 1e3), min_size=4, max_size=128).filter(
    lambda xs: np.linalg.norm(xs) > 1e-8
))
def test_l2_normalize_property(xs):
    v = np.array(xs, dtype=np.float64)
    u = l2_normalize(v)
    assert_unit(u)


# ── conversation_state ─────────────────────────────────────────────────────────

def test_conversation_state_single():
    e = rand_unit()
    state = conversation_state([e])
    assert_unit(state)
    np.testing.assert_allclose(state, e, atol=1e-5)

def test_conversation_state_output_is_unit():
    embeds = [rand_unit() for _ in range(5)]
    state = conversation_state(embeds, decay=0.85)
    assert_unit(state)

def test_conversation_state_recency_bias():
    """Most recent embedding should dominate with low decay."""
    d = 64
    old = np.zeros(d, dtype=np.float32); old[0] = 1.0
    new = np.zeros(d, dtype=np.float32); new[1] = 1.0
    embeds = [l2_normalize(old)] * 4 + [l2_normalize(new)]
    state = conversation_state(embeds, decay=0.5)  # aggressive forgetting
    # newest turn direction should dominate
    assert state[1] > state[0], "Recent embedding should dominate with low decay"

def test_conversation_state_raises_on_empty():
    with pytest.raises((ValueError, IndexError)):
        conversation_state([])


# ── momentum_tangent ───────────────────────────────────────────────────────────

def test_momentum_tangent_output_shapes():
    C_prev = rand_unit()
    C_t = rand_unit()
    M_hat, vel = momentum_tangent(C_t, C_prev, None)
    assert M_hat.shape == C_t.shape
    assert isinstance(vel, float)
    assert vel >= 0.0

def test_momentum_tangent_orthogonal_to_state():
    """Tangent vector must be perpendicular to C_t (tangent plane property)."""
    for _ in range(20):
        C_t = rand_unit()
        C_prev = rand_unit()
        M_hat, vel = momentum_tangent(C_t, C_prev, None)
        if vel > 1e-8:
            dot = float(np.dot(M_hat, C_t))
            assert abs(dot) < 1e-4, f"M_hat not orthogonal to C_t: dot={dot}"

def test_momentum_tangent_unit_when_nonzero():
    """M_hat should be unit when velocity is non-negligible."""
    for _ in range(20):
        C_t = rand_unit()
        C_prev = rand_unit()
        M_hat, vel = momentum_tangent(C_t, C_prev, None)
        if vel > 1e-8:
            assert_unit(M_hat)


# ── predict_future_state ───────────────────────────────────────────────────────

def test_predict_future_state_is_unit():
    """CRITICAL: predicted state must be on unit sphere for valid cosine query."""
    for _ in range(50):
        C_t = rand_unit()
        C_prev = rand_unit()
        M_hat, vel = momentum_tangent(C_t, C_prev, None)
        if vel > 1e-8:
            pred = predict_future_state(C_t, M_hat, vel, k=1)
            assert_unit(pred, tol=1e-4)

@given(st.integers(min_value=1, max_value=5))
def test_predict_future_state_unit_for_all_k(k):
    """Unit sphere invariant holds for any k."""
    C_t = rand_unit(64)
    M_hat = rand_unit(64)
    # Make M_hat orthogonal to C_t (required precondition)
    M_hat = l2_normalize(M_hat - np.dot(M_hat, C_t) * C_t)
    pred = predict_future_state(C_t, M_hat, velocity=0.3, k=k)
    assert_unit(pred, tol=1e-4)

def test_predict_k0_returns_current():
    """k=0 should return C_t itself (no movement)."""
    C_t = rand_unit()
    M_hat = rand_unit()
    M_hat = l2_normalize(M_hat - np.dot(M_hat, C_t) * C_t)
    pred = predict_future_state(C_t, M_hat, velocity=0.5, k=0, step_size=0.3)
    # cos(0)*C_t + sin(0)*M_hat = C_t
    np.testing.assert_allclose(pred, C_t, atol=1e-5)

