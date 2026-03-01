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
        query_vec = self._state.C_t if self._state.C_t is not None else self._state.predict(k=1)
        return [Prediction(
            query_vector=query_vec,
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

