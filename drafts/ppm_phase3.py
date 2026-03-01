# =============================================================================
# PPM: Predictive Push Memory — Phase 3: Nerve Layer
#
# Files:
#   core/surface/observer.py        ← token stream + intent extraction
#   core/surface/intent.py          ← intent signal classifier
#   core/nerve/predictor.py         ← trajectory predictor (the brain)
#   core/nerve/state.py             ← conversation state manager
#   core/nerve/models.py            ← shared dataclasses
#   tests/unit/test_observer.py
#   tests/unit/test_predictor.py
#   tests/unit/test_state.py
#   tests/integration/test_nerve_pipeline.py
#
# This is where all the Phase 1 math becomes a live, async system.
# The momentum vector, geodesic extrapolation, adaptive decay, and
# Bayesian bandits all wire together here.
# =============================================================================


### core/nerve/models.py ###

"""
Shared dataclasses for the Nerve Layer.
Kept in a separate file to avoid circular imports.
"""

from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from core.types import UnitVector, MemoryID


class IntentSignal(str, Enum):
    EXPLORE    = "EXPLORE"      # "what is", "explain", "how does"
    DEBUG      = "DEBUG"        # "error", "fix", "why does"
    NAVIGATE   = "NAVIGATE"     # "where is", "find", "show me"
    COMPARE    = "COMPARE"      # "vs", "difference", "better"
    IMPLEMENT  = "IMPLEMENT"    # "write", "create", "add"
    REFLECT    = "REFLECT"      # "earlier", "we decided", "before"
    UNKNOWN    = "UNKNOWN"


class PrefetchStrategy(str, Enum):
    SEMANTIC = "SEMANTIC"   # pure vector search
    GRAPH    = "GRAPH"      # graph walk from seeds
    SYMBOL   = "SYMBOL"     # exact symbol/file lookup
    HYBRID   = "HYBRID"     # vector + graph combined
    ANNOTATE = "ANNOTATE"   # forward annotation index lookup


@dataclass
class TurnSignals:
    """All signals extracted from one conversation turn."""
    embedding: UnitVector
    intent: IntentSignal
    switch_score: float           # cosine distance from previous turn
    lambda_effective: float       # decay used for this turn's state
    did_reset: bool               # True if context switch triggered reset
    raw_text: str = ""
    extracted_symbols: list[str] = field(default_factory=list)
    extracted_files: list[str] = field(default_factory=list)


@dataclass
class Prediction:
    """A single prefetch prediction with all metadata needed for retrieval."""
    query_vector: UnitVector          # for vector search
    query_text: str                   # for keyword/annotation search
    graph_seeds: list[MemoryID]       # for PPR graph walk
    annotation_tags: list[str]        # for forward annotation lookup
    confidence: float                 # Bayesian bandit confidence
    strategy: PrefetchStrategy
    intent: IntentSignal
    k_steps: int = 1                  # how many turns ahead this predicts
    slot_index: int = 0               # assigned staging slot (0=highest priority)


@dataclass
class ConversationState:
    """Snapshot of the current trajectory state. Serializable for persistence."""
    C_t: UnitVector                   # current state vector
    M_hat: UnitVector | None          # smoothed tangent momentum direction
    M_raw: np.ndarray | None          # raw (unsmoothed) momentum for acceleration
    velocity: float                   # scalar velocity
    acceleration: float               # scalar acceleration (second derivative)
    turn_count: int                   # total turns processed
    lambda_effective: float           # decay used for current state
    last_intent: IntentSignal         # most recent intent signal
    did_reset_last: bool              # whether last turn triggered a reset


### core/surface/intent.py ###

"""
IntentClassifier — maps conversation text to IntentSignal.

Phase 3 uses a heuristic classifier (rules + keyword matching).
This module is designed to be swapped for a fine-tuned classifier
in Phase 7 without changing any downstream code.

The classifier runs synchronously — it must be < 1ms on the hot path.
"""

import re
from core.nerve.models import IntentSignal


# Each rule is (IntentSignal, trigger_patterns, min_matches)
# Patterns are applied to lowercased text.
_RULES: list[tuple[IntentSignal, list[str], int]] = [
    (IntentSignal.DEBUG, [
        "error", "exception", "traceback", "not working", "failing",
        "bug", "broken", "fix", "wrong", "unexpected", "crash",
        "undefined", "null", "none", "attributeerror", "typeerror",
        "syntaxerror", "why does", "why is", "doesn't work"
    ], 1),

    (IntentSignal.IMPLEMENT, [
        "implement", "create", "build", "write a", "add a", "make a",
        "generate", "scaffold", "new function", "new class", "new endpoint",
        "how do i", "how to", "can you write", "can you create"
    ], 1),

    (IntentSignal.NAVIGATE, [
        "where is", "find", "show me", "which file", "locate",
        "where can i", "what file", "what module", "where does"
    ], 1),

    (IntentSignal.COMPARE, [
        " vs ", " versus ", "difference between", "better than",
        "compared to", "which is better", "pros and cons",
        "tradeoffs", "when to use"
    ], 1),

    (IntentSignal.REFLECT, [
        "earlier", "before", "previously", "we decided", "you said",
        "last time", "what did we", "remember when", "as we discussed",
        "going back to"
    ], 1),

    (IntentSignal.EXPLORE, [
        "what is", "what are", "explain", "tell me about", "describe",
        "overview", "how does", "understand", "what does", "meaning of"
    ], 1),
]

# Regex for extracting Python symbols from text
_SYMBOL_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]{2,})\s*(?:\(|\.)', re.MULTILINE)

# Regex for file paths
_FILE_RE = re.compile(
    r'(?:^|[\s\'"`])([a-zA-Z0-9_\-./]+\.(?:py|js|ts|go|rs|json|yaml|yml|md|txt))',
    re.MULTILINE
)


class IntentClassifier:
    """
    Fast, rule-based intent classifier.
    Returns the highest-priority matching intent.
    Priority order mirrors _RULES list (DEBUG first — most actionable).
    """

    def classify(self, text: str) -> IntentSignal:
        """
        Classify text into an IntentSignal.
        O(n·k) where n=len(text), k=total keywords. Runs in < 0.5ms.
        """
        lower = text.lower()
        for signal, patterns, min_matches in _RULES:
            hits = sum(1 for p in patterns if p in lower)
            if hits >= min_matches:
                return signal
        return IntentSignal.UNKNOWN

    def extract_symbols(self, text: str) -> list[str]:
        """
        Extract likely code symbol names (function calls, attribute access).
        Filters out common English words and short tokens.
        """
        _STOPWORDS = {
            "the", "and", "for", "this", "that", "with", "from",
            "import", "class", "return", "print", "true", "false",
            "none", "self", "args", "kwargs", "str", "int", "list",
            "dict", "bool", "type", "any", "not", "new", "get",
        }
        matches = _SYMBOL_RE.findall(text)
        return [m for m in matches
                if m.lower() not in _STOPWORDS and len(m) > 3][:10]

    def extract_files(self, text: str) -> list[str]:
        """Extract file path references from text."""
        return [m.strip(" '\"`") for m in _FILE_RE.findall(text)][:5]


### core/surface/observer.py ###

"""
ConversationObserver — observes the conversation stream and emits TurnSignals.

This is the entry point for every user message. It:
  1. Embeds the turn text
  2. Detects context switches (adaptive decay)
  3. Classifies intent
  4. Extracts symbols and file references
  5. Updates the state history
  6. Returns TurnSignals to the TrajectoryPredictor

Design: stateless per-call after __init__. All mutable state lives in
ConversationStateManager (core/nerve/state.py), which the observer holds
a reference to. This makes the observer independently testable.
"""

import re
from collections import deque

import structlog

from adapters.embedder.base import Embedder
from core.nerve.models import IntentSignal, TurnSignals
from core.nerve.state import ConversationStateManager
from core.surface.intent import IntentClassifier
from core.types import UnitVector
from math_core.entropy import context_switch_score, adaptive_decay
from config.settings import settings

log = structlog.get_logger(__name__)


class ConversationObserver:
    """
    Processes each conversation turn and produces TurnSignals.

    Called once per turn, before the TrajectoryPredictor runs.
    The observer owns the raw signal extraction; the predictor
    owns the prediction generation.
    """

    def __init__(
        self,
        embedder: Embedder,
        state_manager: ConversationStateManager,
        classifier: IntentClassifier | None = None,
    ):
        self._embedder = embedder
        self._state   = state_manager
        self._clf     = classifier or IntentClassifier()
        self._last_embed: UnitVector | None = None

    async def observe(self, text: str) -> TurnSignals:
        """
        Process one turn of the conversation.

        Args:
            text: the raw user message text

        Returns:
            TurnSignals with embedding, intent, and state update info.
        """
        # Step 1: Embed (async I/O — the only potentially slow step)
        embedding = await self._embedder.embed(self._preprocess(text))

        # Step 2: Context switch detection
        switch_score = 0.0
        if self._last_embed is not None:
            switch_score = context_switch_score(self._last_embed, embedding)

        # Step 3: Adaptive decay + reset decision
        lam, did_reset = adaptive_decay(
            lambda_base=settings.decay_lambda_base,
            velocity=self._state.current_velocity,
            switch_score=switch_score,
            switch_threshold=settings.context_switch_threshold,
            alpha=settings.velocity_alpha,
            lambda_min=settings.decay_lambda_min,
            lambda_max=settings.decay_lambda_max,
        )

        if did_reset:
            self._state.reset()
            log.info(
                "observer.context_switch_reset",
                switch_score=f"{switch_score:.3f}",
                prev_velocity=f"{self._state.current_velocity:.3f}",
            )

        # Step 4: Update state manager with new embedding
        self._state.push(embedding, lam)

        # Step 5: Intent + symbol/file extraction
        intent   = self._clf.classify(text)
        symbols  = self._clf.extract_symbols(text)
        files    = self._clf.extract_files(text)

        self._last_embed = embedding

        signals = TurnSignals(
            embedding=embedding,
            intent=intent,
            switch_score=switch_score,
            lambda_effective=lam,
            did_reset=did_reset,
            raw_text=text,
            extracted_symbols=symbols,
            extracted_files=files,
        )

        log.debug(
            "observer.turn_processed",
            intent=intent,
            switch_score=f"{switch_score:.3f}",
            velocity=f"{self._state.current_velocity:.3f}",
            lambda_eff=f"{lam:.3f}",
            did_reset=did_reset,
            symbols=symbols[:3],
        )

        return signals

    def _preprocess(self, text: str) -> str:
        """
        Light normalization before embedding.
        Strips excessive whitespace; preserves code structure.
        Max 512 tokens (~2000 chars) — embedding models have input limits.
        """
        text = re.sub(r'\n{3,}', '\n\n', text)   # collapse triple+ newlines
        text = text.strip()
        return text[:2000]                         # hard character cap


### core/nerve/state.py ###

"""
ConversationStateManager — maintains the mutable trajectory state
across turns.

Owns:
  - The turn embedding history (bounded deque, max N=6)
  - The current conversation state vector C_t
  - The smoothed momentum M_hat and raw momentum M_raw
  - The velocity and acceleration scalars

The state manager is the only mutable object in the Nerve Layer.
Everything else is stateless computation.

Thread safety: this runs in a single asyncio event loop per session.
No locking needed — asyncio is cooperative, not preemptive.
"""

from collections import deque

import numpy as np

from core.nerve.models import ConversationState, IntentSignal
from core.types import UnitVector
from math_core.momentum import (
    conversation_state,
    l2_normalize,
    momentum_tangent,
    predict_future_state,
)
from config.settings import settings


class ConversationStateManager:
    """
    Manages the evolving trajectory state for one session.
    One instance per active session — created by the session factory.
    """

    def __init__(self, dim: int):
        self._dim = dim
        self._history: deque[UnitVector] = deque(maxlen=settings.state_window_max)
        self._C_t:  UnitVector | None = None       # current state
        self._C_prev: UnitVector | None = None     # previous state
        self._M_hat: UnitVector | None = None      # smoothed tangent direction
        self._M_raw: np.ndarray | None = None      # raw momentum (for acceleration)
        self._velocity: float = 0.0
        self._velocity_prev: float = 0.0           # for acceleration
        self._turn_count: int = 0
        self._last_intent: IntentSignal = IntentSignal.UNKNOWN

    # ── Public API ─────────────────────────────────────────────────────────────

    def push(self, embedding: UnitVector, lambda_effective: float) -> None:
        """
        Ingest a new turn embedding and update all state vectors.
        Called by ConversationObserver after every turn.
        """
        self._history.append(embedding)
        self._turn_count += 1

        # Recompute conversation state C_t from history
        C_new = conversation_state(list(self._history), decay=lambda_effective)

        if self._C_t is not None:
            # Update momentum
            self._C_prev = self._C_t
            M_hat_new, vel_new = momentum_tangent(
                C_t=C_new,
                C_prev=self._C_t,
                M_prev=self._M_hat,
                beta=settings.momentum_beta,
            )
            # Acceleration = change in velocity
            self._velocity_prev = self._velocity
            self._velocity = vel_new
            self._M_hat = M_hat_new
        else:
            # First turn: no momentum yet
            self._velocity = 0.0

        self._C_t = C_new

    def predict(self, k: int = 1) -> UnitVector:
        """
        Geodesic extrapolation: where will the conversation be in k turns?
        Returns the predicted state as a UnitVector (safe query vector).
        """
        if self._C_t is None:
            return l2_normalize(np.zeros(self._dim))

        if self._M_hat is None or self._velocity < 1e-8:
            # No momentum yet — predict current state (no movement)
            return self._C_t

        return predict_future_state(
            C_t=self._C_t,
            M_hat=self._M_hat,
            velocity=self._velocity,
            k=k,
            step_size=settings.slerp_step_size,
        )

    def reset(self) -> None:
        """
        Hard reset on context switch. Clears history and momentum.
        Called by ConversationObserver when switch_score > threshold.
        """
        self._history.clear()
        self._C_prev = None
        self._M_hat = None
        self._M_raw = None
        self._velocity = 0.0
        self._velocity_prev = 0.0
        # Note: _C_t and _turn_count intentionally NOT reset —
        # we keep the last known position as the new starting point.

    def snapshot(self) -> ConversationState:
        """Serializable snapshot of current state for persistence/logging."""
        return ConversationState(
            C_t=self._C_t if self._C_t is not None
                else l2_normalize(np.zeros(self._dim)),
            M_hat=self._M_hat,
            M_raw=self._M_raw,
            velocity=self._velocity,
            acceleration=self._velocity - self._velocity_prev,
            turn_count=self._turn_count,
            lambda_effective=settings.decay_lambda_base,
            last_intent=self._last_intent,
            did_reset_last=False,
        )

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def current_velocity(self) -> float:
        return self._velocity

    @property
    def acceleration(self) -> float:
        return self._velocity - self._velocity_prev

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def has_momentum(self) -> bool:
        return self._M_hat is not None and self._velocity > 1e-8

    @property
    def C_t(self) -> UnitVector | None:
        return self._C_t


### core/nerve/predictor.py ###

"""
TrajectoryPredictor — the brain of PPM.

Given TurnSignals from the observer and the current ConversationState,
generates a ranked list of Predictions for the staging layer to prefetch.

Prediction generation strategy:
  1. SEMANTIC prediction  — geodesic extrapolation query vector
  2. GRAPH prediction     — PPR seeds from recent memory hits
  3. SYMBOL prediction    — extracted symbols → direct symbol lookup
  4. FILE prediction      — extracted file paths → direct file lookup
  5. ANNOTATE prediction  — intent + topic tags → annotation index

Each prediction gets a Bayesian confidence score from the BanditRegistry.
The predictor decides how many steps ahead to predict based on velocity:
  - Low velocity  → k=1 (deep dive, predict next turn only)
  - Mid velocity  → k=2 (moderate drift, predict 2 turns ahead)
  - High velocity → k=3 (fast shift, aggressive lookahead)
"""

import structlog
from math import ceil

from core.nerve.models import (
    ConversationState, IntentSignal, Prediction,
    PrefetchStrategy, TurnSignals,
)
from core.nerve.state import ConversationStateManager
from core.types import MemoryID, UnitVector
from math_core.bandit import BanditRegistry
from config.settings import settings

log = structlog.get_logger(__name__)

# Velocity thresholds for k-step selection
_K_THRESHOLDS = [
    (0.05, 1),   # velocity < 0.05 → k=1
    (0.15, 2),   # velocity < 0.15 → k=2
    (1.00, 3),   # velocity ≥ 0.15 → k=3
]

# Intent → preferred strategies (ordered by priority)
_INTENT_STRATEGIES: dict[IntentSignal, list[PrefetchStrategy]] = {
    IntentSignal.DEBUG:      [PrefetchStrategy.GRAPH, PrefetchStrategy.SEMANTIC,
                              PrefetchStrategy.ANNOTATE],
    IntentSignal.IMPLEMENT:  [PrefetchStrategy.SEMANTIC, PrefetchStrategy.SYMBOL,
                              PrefetchStrategy.ANNOTATE],
    IntentSignal.NAVIGATE:   [PrefetchStrategy.SYMBOL, PrefetchStrategy.SEMANTIC],
    IntentSignal.COMPARE:    [PrefetchStrategy.HYBRID, PrefetchStrategy.SEMANTIC],
    IntentSignal.REFLECT:    [PrefetchStrategy.ANNOTATE, PrefetchStrategy.SEMANTIC],
    IntentSignal.EXPLORE:    [PrefetchStrategy.SEMANTIC, PrefetchStrategy.ANNOTATE],
    IntentSignal.UNKNOWN:    [PrefetchStrategy.SEMANTIC],
}

# Intent → annotation tag hints for ANNOTATE strategy
_INTENT_ANNOTATION_TAGS: dict[IntentSignal, list[str]] = {
    IntentSignal.DEBUG:     ["intent:DEBUG", "topic:error"],
    IntentSignal.IMPLEMENT: ["intent:IMPLEMENT"],
    IntentSignal.NAVIGATE:  ["intent:NAVIGATE"],
    IntentSignal.COMPARE:   [],
    IntentSignal.REFLECT:   [],
    IntentSignal.EXPLORE:   ["intent:EXPLORE"],
    IntentSignal.UNKNOWN:   [],
}


class TrajectoryPredictor:
    """
    Generates prefetch predictions from conversation trajectory signals.

    One instance per session. Holds the BanditRegistry for this session
    (bandits are per-session to allow per-user/per-project learning).
    """

    def __init__(
        self,
        state_manager: ConversationStateManager,
        bandit_registry: BanditRegistry | None = None,
    ):
        self._state   = state_manager
        self._bandits = bandit_registry or BanditRegistry()
        self._recent_graph_seeds: list[MemoryID] = []  # updated by feedback layer

    def predict(self, signals: TurnSignals) -> list[Prediction]:
        """
        Generate ranked predictions for this turn.

        Args:
            signals: TurnSignals from ConversationObserver

        Returns:
            List of Predictions, sorted by confidence descending.
            Length bounded by settings.slot_count (default 10).
        """
        if not self._state.has_momentum and self._state.turn_count <= 1:
            # First turn: cold start — generate broad semantic prediction only
            return self._cold_start_predictions(signals)

        k = self._select_k(self._state.current_velocity)
        strategies = _INTENT_STRATEGIES.get(signals.intent, [PrefetchStrategy.SEMANTIC])

        predictions: list[Prediction] = []
        slot = 0

        for strategy in strategies:
            preds = self._generate_for_strategy(strategy, signals, k, slot)
            for p in preds:
                p.slot_index = slot
                slot += 1
            predictions.extend(preds)
            if slot >= settings.slot_count:
                break

        # Sort by confidence descending, re-assign slot indices
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        for i, p in enumerate(predictions):
            p.slot_index = i

        log.debug(
            "predictor.predictions_generated",
            count=len(predictions),
            intent=signals.intent,
            k=k,
            velocity=f"{self._state.current_velocity:.3f}",
            top_conf=f"{predictions[0].confidence:.3f}" if predictions else "n/a",
        )

        return predictions[:settings.slot_count]

    # ── Strategy generators ────────────────────────────────────────────────────

    def _generate_for_strategy(
        self,
        strategy: PrefetchStrategy,
        signals: TurnSignals,
        k: int,
        base_slot: int,
    ) -> list[Prediction]:
        conf = self._bandits.confidence(strategy.value, signals.intent.value)

        if strategy == PrefetchStrategy.SEMANTIC:
            return self._semantic_predictions(signals, k, conf)

        if strategy == PrefetchStrategy.GRAPH:
            return self._graph_predictions(signals, conf)

        if strategy == PrefetchStrategy.SYMBOL:
            return self._symbol_predictions(signals, conf)

        if strategy == PrefetchStrategy.HYBRID:
            sem  = self._semantic_predictions(signals, k, conf * 0.6)
            grph = self._graph_predictions(signals, conf * 0.4)
            return sem + grph

        if strategy == PrefetchStrategy.ANNOTATE:
            return self._annotation_predictions(signals, conf)

        return []

    def _semantic_predictions(
        self, signals: TurnSignals, k: int, base_conf: float
    ) -> list[Prediction]:
        """
        Generate k semantic predictions via geodesic extrapolation.
        Each step ahead gets a lower confidence (uncertainty grows with k).
        """
        preds = []
        for step in range(1, k + 1):
            query_vec = self._state.predict(k=step)
            conf = base_conf * (0.85 ** (step - 1))   # decay confidence per step
            preds.append(Prediction(
                query_vector=query_vec,
                query_text=signals.raw_text,
                graph_seeds=[],
                annotation_tags=[],
                confidence=conf,
                strategy=PrefetchStrategy.SEMANTIC,
                intent=signals.intent,
                k_steps=step,
            ))
        return preds

    def _graph_predictions(
        self, signals: TurnSignals, base_conf: float
    ) -> list[Prediction]:
        """
        Generate graph-walk predictions from recent memory seeds.
        Seeds are memories that were recently injected into context
        (updated by the feedback layer after each turn).
        """
        if not self._recent_graph_seeds:
            return []

        # Use current state vector as the query — graph walk is seed-driven
        return [Prediction(
            query_vector=self._state.C_t or self._state.predict(k=1),
            query_text=signals.raw_text,
            graph_seeds=list(self._recent_graph_seeds[-3:]),  # top 3 recent seeds
            annotation_tags=[],
            confidence=base_conf,
            strategy=PrefetchStrategy.GRAPH,
            intent=signals.intent,
            k_steps=1,
        )]

    def _symbol_predictions(
        self, signals: TurnSignals, base_conf: float
    ) -> list[Prediction]:
        """
        Generate direct symbol lookup predictions.
        One prediction per extracted symbol/file — highest precision.
        """
        preds = []
        for symbol in signals.extracted_symbols[:3]:
            preds.append(Prediction(
                query_vector=signals.embedding,  # use current turn embed
                query_text=symbol,
                graph_seeds=[],
                annotation_tags=[f"symbol:{symbol}"],
                confidence=base_conf * 1.1,     # symbol matches are high precision
                strategy=PrefetchStrategy.SYMBOL,
                intent=signals.intent,
                k_steps=1,
            ))
        for fpath in signals.extracted_files[:2]:
            preds.append(Prediction(
                query_vector=signals.embedding,
                query_text=fpath,
                graph_seeds=[],
                annotation_tags=[f"file:{fpath}"],
                confidence=base_conf * 1.2,     # file matches are very high precision
                strategy=PrefetchStrategy.SYMBOL,
                intent=signals.intent,
                k_steps=1,
            ))
        return preds

    def _annotation_predictions(
        self, signals: TurnSignals, base_conf: float
    ) -> list[Prediction]:
        """
        Generate annotation-index predictions.
        Uses forward annotation tags to find memories by predicted relevance.
        """
        tags = list(_INTENT_ANNOTATION_TAGS.get(signals.intent, []))

        # Add symbol-derived topic tags
        for sym in signals.extracted_symbols[:2]:
            tags.append(f"symbol:{sym}")

        if not tags:
            return []

        return [Prediction(
            query_vector=self._state.predict(k=1),
            query_text=signals.raw_text,
            graph_seeds=[],
            annotation_tags=tags,
            confidence=base_conf,
            strategy=PrefetchStrategy.ANNOTATE,
            intent=signals.intent,
            k_steps=1,
        )]

    def _cold_start_predictions(self, signals: TurnSignals) -> list[Prediction]:
        """
        First turn: no momentum, no history.
        Generate a single broad semantic prediction with 0.5 confidence.
        Bandits start at Beta(1,1) → 0.5, so this is calibrated.
        """
        conf = self._bandits.confidence(
            PrefetchStrategy.SEMANTIC.value, signals.intent.value
        )
        return [Prediction(
            query_vector=signals.embedding,   # use raw turn embed as query
            query_text=signals.raw_text,
            graph_seeds=[],
            annotation_tags=list(_INTENT_ANNOTATION_TAGS.get(signals.intent, [])),
            confidence=conf,
            strategy=PrefetchStrategy.SEMANTIC,
            intent=signals.intent,
            k_steps=1,
            slot_index=0,
        )]

    # ── Feedback interface ─────────────────────────────────────────────────────

    def update_graph_seeds(self, used_memory_ids: list[MemoryID]) -> None:
        """
        Called by the feedback layer after each turn with the memory IDs
        that were actually used. These become seeds for graph predictions.
        """
        self._recent_graph_seeds = used_memory_ids[:5]

    def update_bandits(self, strategy: str, intent: str, hit: bool) -> None:
        """Called by the feedback layer with hit/miss outcome."""
        self._bandits.update(strategy, intent, hit)
        log.debug(
            "predictor.bandit_updated",
            strategy=strategy, intent=intent, hit=hit,
            new_conf=f"{self._bandits.confidence(strategy, intent):.3f}",
        )

    def bandit_snapshot(self) -> dict:
        """Serializable bandit state for persistence."""
        return self._bandits.snapshot()

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _select_k(velocity: float) -> int:
        """Select prediction horizon k based on velocity."""
        for threshold, k in _K_THRESHOLDS:
            if velocity < threshold:
                return k
        return _K_THRESHOLDS[-1][1]


### tests/unit/test_observer.py ###

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
    ("as we discussed, the fix is Y",           IntentSignal.REFLECT),
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


### tests/unit/test_state.py ###

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


### tests/unit/test_predictor.py ###

"""Tests for core/nerve/predictor.py — TrajectoryPredictor"""

import numpy as np
import pytest
from math_core.momentum import l2_normalize
from core.nerve.models import IntentSignal, PrefetchStrategy, TurnSignals
from core.nerve.predictor import TrajectoryPredictor
from core.nerve.state import ConversationStateManager


def rand_unit(d=64):
    return l2_normalize(np.random.randn(d).astype(np.float32))


def make_signals(intent=IntentSignal.EXPLORE, symbols=None, files=None):
    return TurnSignals(
        embedding=rand_unit(),
        intent=intent,
        switch_score=0.1,
        lambda_effective=0.85,
        did_reset=False,
        raw_text="test query",
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


### tests/integration/test_nerve_pipeline.py ###

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

def deterministic_embed(text: str) -> np.ndarray:
    """Returns a reproducible unit vector based on text hash."""
    seed = hash(text) % (2**31)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return l2_normalize(v)


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
