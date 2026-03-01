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

