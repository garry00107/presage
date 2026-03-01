"""
FeedbackLoop — closes the loop between feedback and the prediction system.

After every turn, FeedbackLoop takes a TurnFeedback and:
  1. Updates Bayesian bandits in the TrajectoryPredictor
  2. Updates graph seeds in the TrajectoryPredictor (for GRAPH strategy)
  3. Records a TrajectorySample in the dataset
  4. Updates forward annotations in MetaStore (memories tag themselves)
  5. Logs observability metrics

This is the component that makes PPM self-improving over time.
The predictor gets smarter every turn — no retraining needed.
"""

import structlog

from core.feedback.dataset import TrajectoryDataset
from core.feedback.models import TurnFeedback
from core.nerve.predictor import TrajectoryPredictor
from core.nerve.state import ConversationStateManager

log = structlog.get_logger(__name__)


class FeedbackLoop:
    """
    Closes the prediction → feedback → improvement cycle.
    Stateless per-call — all state lives in predictor and dataset.
    """

    def __init__(
        self,
        predictor: TrajectoryPredictor,
        state: ConversationStateManager,
        dataset: TrajectoryDataset,
        meta_store,      # MetaStore — avoid circular import
        annotation_hit_threshold: float = 0.50,
    ):
        self._predictor   = predictor
        self._state       = state
        self._dataset     = dataset
        self._meta        = meta_store
        self._ann_threshold = annotation_hit_threshold

    async def process(
        self,
        feedback: TurnFeedback,
        switch_score: float = 0.0,
    ) -> None:
        """
        Process TurnFeedback and update all downstream systems.

        Args:
            feedback:     TurnFeedback from FeedbackTracker
            switch_score: context switch score for this turn (for dataset)

        This method is non-blocking on the hot path — runs after
        the LLM response has been sent to the user.
        """
        if not feedback.results:
            return

        # Step 1: Update Bayesian bandits
        self._update_bandits(feedback)

        # Step 2: Update graph seeds from hit memory IDs
        if feedback.used_memory_ids:
            self._predictor.update_graph_seeds(feedback.used_memory_ids)

        # Step 3: Record trajectory sample (async, non-blocking)
        await self._dataset.record(feedback, self._state, switch_score)

        # Step 4: Update forward annotations for hit memories
        await self._update_annotations(feedback)

        # Step 5: Log per-turn observability
        self._log_metrics(feedback)

    # ── Internal steps ─────────────────────────────────────────────────────────

    def _update_bandits(self, feedback: TurnFeedback) -> None:
        """
        Update Bayesian bandits for every prediction in this turn.
        One update per (strategy, intent) pair per staged memory.
        """
        for result in feedback.results:
            self._predictor.update_bandits(
                strategy=result.strategy.value,
                intent=result.intent.value,
                hit=result.is_hit,
            )

        log.debug(
            "feedback_loop.bandits_updated",
            turn=feedback.turn_index,
            updates=len(feedback.results),
        )

    async def _update_annotations(self, feedback: TurnFeedback) -> None:
        """
        When a memory is a hit, increment its annotation hit_count
        and add new forward annotation tags based on current intent.

        This implements the "memories tag themselves with future relevance"
        principle: if auth.js was useful during a DEBUG:authentication turn,
        it gets tagged intent:DEBUG and topic:auth with higher weight.
        """
        if not feedback.used_memory_ids:
            return

        # Current intent → annotation tag to add
        intent_tag = f"intent:{feedback.intent.value}"

        for memory_id in feedback.used_memory_ids:
            # Increment hit count on all existing annotations for this memory
            try:
                await self._meta.increment_annotation_hit(memory_id, intent_tag)
            except Exception:
                pass  # annotation may not exist yet — silently skip

            # Add the current intent as a new forward annotation if not present
            try:
                await self._meta._db.execute(
                    """INSERT OR IGNORE INTO forward_annotations
                       (memory_id, context_tag, weight, created_at, hit_count)
                       VALUES (?, ?, ?, strftime('%s','now'), 1)""",
                    (memory_id, intent_tag, 1.0)
                )
                await self._meta._db.commit()
            except Exception as e:
                log.warning(
                    "feedback_loop.annotation_update_failed",
                    memory_id=memory_id,
                    error=str(e),
                )

    def _log_metrics(self, feedback: TurnFeedback) -> None:
        """Structured log for observability dashboards."""
        hit_signals = {}
        for r in feedback.results:
            if r.is_hit:
                hit_signals[r.hit_signal] = hit_signals.get(r.hit_signal, 0) + 1

        log.info(
            "feedback_loop.turn_complete",
            session=feedback.session_id,
            turn=feedback.turn_index,
            intent=feedback.intent.value,
            hit_rate=f"{feedback.hit_rate:.2f}",
            total_staged=feedback.total_staged,
            total_hits=feedback.total_hits,
            hit_signals=hit_signals,
            used_memory_count=len(feedback.used_memory_ids),
        )

