"""
FeedbackTracker — orchestrates per-turn feedback collection.

Called once per turn AFTER the LLM response is generated.
Coordinates between:
  - StagingCache  (what was staged and injected)
  - HitMissDetector (was it used?)
  - FeedbackLoop  (update bandits, seeds, annotations)

The tracker is the single entry point for the feedback layer.
Everything downstream is triggered by tracker.evaluate_turn().
"""

import asyncio
import uuid
import structlog

from core.feedback.detector import HitMissDetector
from core.feedback.models import TurnFeedback
from core.staging.cache import StagingCache
from core.staging.models import StagedMemory
from core.types import MemoryID, UnitVector
from adapters.embedder.base import Embedder

log = structlog.get_logger(__name__)


class FeedbackTracker:
    """
    Per-session feedback tracker.
    One instance per session — holds references to session-specific objects.
    """

    def __init__(
        self,
        cache: StagingCache,
        detector: HitMissDetector,
        embedder: Embedder,
        session_id: str,
    ):
        self._cache      = cache
        self._detector   = detector
        self._embedder   = embedder
        self._session_id = session_id
        self._turn_index = 0

    async def evaluate_turn(
        self,
        response_text: str,
        intent,                         # IntentSignal
        memory_embeddings: dict | None = None,
    ) -> TurnFeedback:
        """
        Evaluate the feedback for one completed turn.

        Args:
            response_text:      the full LLM response text for this turn
            intent:             IntentSignal detected for this turn
            memory_embeddings:  optional {memory_id: UnitVector} for semantic detection

        Returns:
            TurnFeedback with all hit/miss results and aggregates.
        """
        self._turn_index += 1
        turn_id = str(uuid.uuid4())

        # Step 1: Get all staged memories that were injected this turn
        injected = await self._cache.drain_for_feedback()

        if not injected:
            log.debug(
                "feedback_tracker.no_staged_memories",
                turn=self._turn_index,
            )
            return TurnFeedback(
                turn_id=turn_id,
                session_id=self._session_id,
                turn_index=self._turn_index,
                intent=intent,
                results=[],
            )

        # Step 2: Embed the response (async — may take ~50ms)
        response_embedding: UnitVector = await self._embedder.embed(
            response_text[:2000]   # cap for embedding model limits
        )

        # Step 3: Detect hits/misses for each staged memory
        results = self._detector.detect_batch(
            staged_memories=injected,
            response_text=response_text,
            response_embedding=response_embedding,
            memory_embeddings=memory_embeddings,
        )

        # Step 4: Mark hits in the cache (for graph seed updates)
        hit_ids = [r.memory_id for r in results if r.is_hit]
        if hit_ids:
            await self._cache.mark_used(hit_ids)

        # Step 5: Build TurnFeedback
        feedback = TurnFeedback(
            turn_id=turn_id,
            session_id=self._session_id,
            turn_index=self._turn_index,
            intent=intent,
            results=results,
        )
        feedback.compute_aggregates()

        log.info(
            "feedback_tracker.turn_evaluated",
            turn=self._turn_index,
            staged=feedback.total_staged,
            hits=feedback.total_hits,
            hit_rate=f"{feedback.hit_rate:.2f}",
            intent=intent.value,
        )

        return feedback

