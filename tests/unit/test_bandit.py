"""Tests for math/bandit.py"""

import pytest
from math_core.bandit import BetaBandit, BanditRegistry


def test_initial_confidence():
    b = BetaBandit()
    assert abs(b.confidence() - 0.5) < 1e-9

def test_update_hit_increases_confidence():
    b = BetaBandit()
    for _ in range(10):
        b.update(hit=True)
    assert b.confidence() > 0.5

def test_update_miss_decreases_confidence():
    b = BetaBandit()
    for _ in range(10):
        b.update(hit=False)
    assert b.confidence() < 0.5

def test_n_observations():
    b = BetaBandit()
    assert b.n_observations() == 0
    b.update(True); b.update(False)
    assert b.n_observations() == 2

def test_sample_range():
    b = BetaBandit()
    for _ in range(100):
        s = b.sample()
        assert 0.0 <= s <= 1.0

def test_uncertainty_decreases_with_data():
    b = BetaBandit()
    u0 = b.uncertainty()
    for _ in range(100):
        b.update(True)
    assert b.uncertainty() < u0

def test_registry_creates_bandits():
    reg = BanditRegistry()
    b1 = reg.get("SEMANTIC", "DEBUG")
    b2 = reg.get("GRAPH", "DEBUG")
    assert b1 is not b2

def test_registry_snapshot_roundtrip():
    reg = BanditRegistry()
    reg.update("SEMANTIC", "DEBUG", hit=True)
    reg.update("GRAPH", "EXPLORE", hit=False)
    snap = reg.snapshot()
    reg2 = BanditRegistry.from_snapshot(snap)
    assert abs(reg2.confidence("SEMANTIC", "DEBUG") -
               reg.confidence("SEMANTIC", "DEBUG")) < 1e-9
