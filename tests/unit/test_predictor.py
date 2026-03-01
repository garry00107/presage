"""Tests for core/nerve/predictor.py — TrajectoryPredictor"""

import numpy as np
import pytest
from math_core.momentum import l2_normalize
from core.nerve.models import IntentSignal, PrefetchStrategy, TurnSignals
from core.nerve.predictor import TrajectoryPredictor
from core.nerve.state import ConversationStateManager


def rand_unit(d=64):
    return l2_normalize(np.random.randn(d).astype(np.float32))


def make_signals(text="test query", intent=IntentSignal.EXPLORE, symbols=None, files=None) -> TurnSignals:
    if files:
        intent = IntentSignal.NAVIGATE
    return TurnSignals(
        embedding=rand_unit(),
        intent=intent,
        switch_score=0.1,
        lambda_effective=0.85,
        did_reset=False,
        raw_text=text,
        extracted_symbols=symbols or [],
        extracted_files=files or [],
    )


def make_predictor(n_turns=3, dim=64):
    state = ConversationStateManager(dim=dim)
    for _ in range(n_turns):
        state.push(rand_unit(dim), 0.85)
    return TrajectoryPredictor(state), state


def test_predict_returns_list():
    pred, _ = make_predictor()
    preds = pred.predict(make_signals())
    assert isinstance(preds, list)

def test_cold_start_single_prediction():
    state = ConversationStateManager(dim=64)  # no turns pushed
    predictor = TrajectoryPredictor(state)
    preds = predictor.predict(make_signals())
    assert len(preds) >= 1
    assert preds[0].strategy == PrefetchStrategy.SEMANTIC

def test_predictions_sorted_by_confidence():
    pred, _ = make_predictor(n_turns=5)
    preds = pred.predict(make_signals())
    confs = [p.confidence for p in preds]
    assert confs == sorted(confs, reverse=True)

def test_slot_indices_sequential():
    pred, _ = make_predictor()
    preds = pred.predict(make_signals())
    for i, p in enumerate(preds):
        assert p.slot_index == i

def test_prediction_vectors_are_unit():
    pred, _ = make_predictor(n_turns=5)
    preds = pred.predict(make_signals())
    for p in preds:
        norm = np.linalg.norm(p.query_vector)
        assert abs(norm - 1.0) < 1e-4, f"Query vector not unit: norm={norm}"

def test_debug_intent_generates_graph_prediction():
    pred, _ = make_predictor()
    pred.update_graph_seeds(["mem-001", "mem-002"])
    preds = pred.predict(make_signals(intent=IntentSignal.DEBUG))
    strategies = {p.strategy for p in preds}
    assert PrefetchStrategy.GRAPH in strategies

def test_symbol_intent_generates_symbol_prediction():
    pred, _ = make_predictor()
    preds = pred.predict(make_signals(
        intent=IntentSignal.NAVIGATE,
        symbols=["verify_token", "refresh_session"]
    ))
    strategies = {p.strategy for p in preds}
    assert PrefetchStrategy.SYMBOL in strategies

def test_symbol_predictions_have_annotation_tags():
    pred, _ = make_predictor()
    preds = pred.predict(make_signals(symbols=["my_function"]))
    sym_preds = [p for p in preds if p.strategy == PrefetchStrategy.SYMBOL]
    if sym_preds:
        assert any("symbol:my_function" in p.annotation_tags for p in sym_preds)

def test_file_predictions_generated():
    pred, _ = make_predictor()
    preds = pred.predict(make_signals(files=["src/auth.py"]))
    file_preds = [p for p in preds if "file:src/auth.py" in p.annotation_tags]
    assert len(file_preds) >= 1

def test_max_slot_count_respected():
    from config.settings import settings
    pred, _ = make_predictor(n_turns=5)
    preds = pred.predict(make_signals())
    assert len(preds) <= settings.slot_count

def test_bandit_update_changes_confidence():
    pred, _ = make_predictor()
    key_s, key_i = PrefetchStrategy.SEMANTIC.value, IntentSignal.EXPLORE.value
    conf_before = pred._bandits.confidence(key_s, key_i)
    for _ in range(10):
        pred.update_bandits(key_s, key_i, hit=True)
    conf_after = pred._bandits.confidence(key_s, key_i)
    assert conf_after > conf_before

def test_graph_seeds_update():
    pred, _ = make_predictor()
    pred.update_graph_seeds(["m1", "m2", "m3"])
    assert "m1" in pred._recent_graph_seeds

def test_bandit_snapshot_roundtrip():
    pred, _ = make_predictor()
    pred.update_bandits("SEMANTIC", "DEBUG", hit=True)
    snap = pred.bandit_snapshot()
    assert "SEMANTIC:DEBUG" in snap

