"""Tests for core/surface/observer.py and core/surface/intent.py"""

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.nerve.models import IntentSignal
from core.nerve.state import ConversationStateManager
from core.surface.intent import IntentClassifier
from core.surface.observer import ConversationObserver
from math_core.momentum import l2_normalize


# ── IntentClassifier ───────────────────────────────────────────────────────────

clf = IntentClassifier()


@pytest.mark.parametrize("text,expected", [
    ("why does my code throw an error?",        IntentSignal.DEBUG),
    ("there's an exception in the auth module", IntentSignal.DEBUG),
    ("write a function to validate tokens",     IntentSignal.IMPLEMENT),
    ("create a new endpoint for users",         IntentSignal.IMPLEMENT),
    ("where is the auth middleware defined?",   IntentSignal.NAVIGATE),
    ("find the User model",                     IntentSignal.NAVIGATE),
    ("difference between JWT and sessions",     IntentSignal.COMPARE),
    ("what is dependency injection?",           IntentSignal.EXPLORE),
    ("explain how async works",                 IntentSignal.EXPLORE),
    ("earlier you mentioned we should use X",   IntentSignal.REFLECT),
    ("as we discussed, the solution is Y",      IntentSignal.REFLECT),
    ("blah blah xyz random nonsense",           IntentSignal.UNKNOWN),
])
def test_intent_classification(text, expected):
    assert clf.classify(text) == expected


def test_symbol_extraction():
    text = "Can you fix the verify_token() function and check authenticate.middleware?"
    symbols = clf.extract_symbols(text)
    assert "verify_token" in symbols

def test_symbol_extraction_filters_stopwords():
    text = "use the get() method and check self.type"
    symbols = clf.extract_symbols(text)
    assert "get" not in symbols
    assert "self" not in symbols

def test_file_extraction():
    text = "Look at src/auth/login.py and also config/settings.yaml"
    files = clf.extract_files(text)
    assert any("login.py" in f for f in files)
    assert any("settings.yaml" in f for f in files)

def test_file_extraction_no_false_positives():
    text = "There is no file path in this sentence at all."
    files = clf.extract_files(text)
    assert files == []


# ── ConversationObserver ───────────────────────────────────────────────────────

def make_observer(dim=64):
    """Build observer with mock embedder."""
    embedder = AsyncMock()
    embedder.embed = AsyncMock(
        side_effect=lambda text: l2_normalize(np.random.randn(dim).astype(np.float32))
    )
    state_mgr = ConversationStateManager(dim=dim)
    return ConversationObserver(embedder, state_mgr), embedder, state_mgr


@pytest.mark.asyncio
async def test_observer_returns_turn_signals():
    obs, _, _ = make_observer()
    signals = await obs.observe("why does my function crash?")
    assert signals.intent == IntentSignal.DEBUG
    assert signals.embedding is not None
    assert abs(np.linalg.norm(signals.embedding) - 1.0) < 1e-5

@pytest.mark.asyncio
async def test_observer_first_turn_no_reset():
    obs, _, _ = make_observer()
    signals = await obs.observe("hello world")
    assert signals.did_reset is False
    assert signals.switch_score == 0.0

@pytest.mark.asyncio
async def test_observer_calls_embedder():
    obs, embedder, _ = make_observer()
    await obs.observe("test message")
    embedder.embed.assert_called_once()

@pytest.mark.asyncio
async def test_observer_updates_state():
    obs, _, state_mgr = make_observer()
    await obs.observe("first turn")
    assert state_mgr.turn_count == 1
    await obs.observe("second turn")
    assert state_mgr.turn_count == 2

@pytest.mark.asyncio
async def test_observer_detects_context_switch():
    """
    Simulate a context switch by patching context_switch_score to return
    a value above the threshold.
    """
    import core.surface.observer as obs_module
    obs, _, state_mgr = make_observer()

    # First turn — establishes last_embed
    await obs.observe("debugging authentication")

    # Monkey-patch to force a context switch on next turn
    original = obs_module.context_switch_score
    obs_module.context_switch_score = lambda a, b: 0.99  # above threshold

    signals = await obs.observe("completely different topic")
    assert signals.did_reset is True
    assert signals.switch_score == 0.99

    obs_module.context_switch_score = original  # restore

@pytest.mark.asyncio
async def test_observer_extracts_symbols_and_files():
    obs, _, _ = make_observer()
    signals = await obs.observe(
        "Fix the verify_token() function in src/auth/tokens.py"
    )
    assert "verify_token" in signals.extracted_symbols
    assert any("tokens.py" in f for f in signals.extracted_files)

