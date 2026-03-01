"""
SessionManager — the central coordinator for one PPM session.

A session represents one continuous conversation with one LLM context.
The SessionManager owns the lifecycle of all phase components for
this session and exposes a single clean interface:

    result = await session.turn(user_message)

Internally this orchestrates:
  Phase 3: Observer  → extract signals from user message
  Phase 3: Predictor → generate predictions from trajectory
  Phase 4: Cache     → schedule background prefetch (non-blocking)
  Phase 4: Injector  → select memories to inject into context
  Phase 4: Reranker  → refine staged memory relevance
  LLM:     call with enriched context
  Phase 5: Tracker   → evaluate hit/miss for this turn
  Phase 5: Loop      → update bandits, seeds, dataset
  Phase 2: Pipeline  → distill and write new memories

The hot path (user sees latency) is:
  observe → inject → LLM → stream response

Everything else runs after the response starts streaming.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field

import structlog

from adapters.embedder.base import Embedder
from core.feedback.detector import HitMissDetector
from core.feedback.loop import FeedbackLoop
from core.feedback.tracker import FeedbackTracker
from core.nerve.models import IntentSignal
from core.nerve.predictor import TrajectoryPredictor
from core.nerve.state import ConversationStateManager
from core.staging.cache import StagingCache
from core.staging.injector import Injector
from core.staging.models import InjectionPlan
from core.staging.reranker import Reranker
from core.surface.observer import ConversationObserver
from core.write.pipeline import WritePipeline
from config.settings import settings

log = structlog.get_logger(__name__)


@dataclass
class TurnResult:
    """The complete result of processing one conversation turn."""
    session_id: str
    turn_index: int
    user_message: str
    llm_response: str
    injection_plan: InjectionPlan
    intent: IntentSignal
    velocity: float
    hit_rate: float
    tokens_injected: int
    memories_injected: int
    latency_ms: float
    staged_slot_summary: list[dict] = field(default_factory=list)


class SessionManager:
    """
    Manages one PPM session end-to-end.

    Lifecycle:
      1. Created by SessionFactory
      2. turn() called for each user message
      3. close() called when session ends (persists bandit state)
    """

    def __init__(
        self,
        session_id: str,
        observer: ConversationObserver,
        predictor: TrajectoryPredictor,
        state: ConversationStateManager,
        cache: StagingCache,
        injector: Injector,
        reranker: Reranker,
        tracker: FeedbackTracker,
        feedback_loop: FeedbackLoop,
        write_pipeline: WritePipeline,
        embedder: Embedder,
        llm_caller,     # callable: async (prompt, context) → str
        meta_store,
    ):
        self.session_id    = session_id
        self._observer     = observer
        self._predictor    = predictor
        self._state        = state
        self._cache        = cache
        self._injector     = injector
        self._reranker     = reranker
        self._tracker      = tracker
        self._loop         = feedback_loop
        self._write        = write_pipeline
        self._embedder     = embedder
        self._llm          = llm_caller
        self._meta         = meta_store
        self._turn_index   = 0
        self._last_switch_score = 0.0

    async def turn(
        self,
        user_message: str,
        source: str = "",
        stream: bool = False,
    ) -> TurnResult:
        """
        Process one conversation turn end-to-end.

        Hot path (blocks until response):
          1. Observe user message → TurnSignals
          2. Get AUTO-tier staged memories (already prefetched)
          3. Rerank against actual message embedding
          4. Knapsack inject into context
          5. Call LLM with enriched context
          6. Return response

        Post-response (async, non-blocking):
          7. Schedule next prefetch
          8. Run feedback (hit/miss detection)
          9. Distill and write new memories
          10. Update bandits + graph seeds

        Args:
            user_message: raw user input
            source:       optional source identifier (file path, url)
            stream:       if True, return async generator (Phase 7)

        Returns:
            TurnResult with response and all metadata
        """
        t0 = time.monotonic()
        self._turn_index += 1

        log.info(
            "session.turn_start",
            session=self.session_id,
            turn=self._turn_index,
            message_len=len(user_message),
        )

        # ── HOT PATH ──────────────────────────────────────────────────────────

        # Step 1: Observe — embed + intent + signals
        signals = await self._observer.observe(user_message)
        self._last_switch_score = signals.switch_score

        # Step 2: Get AUTO-tier staged memories (ready from prev turn's prefetch)
        await self._cache.evict_expired()
        staged = await self._cache.get_auto_inject()

        # Also get HOT-tier if soft trigger fires
        hot = await self._cache.get_hot(trigger_text=user_message)
        staged = self._deduplicate_staged(staged + hot)

        # Step 3: Rerank against actual message embedding
        if staged:
            chunk_embeddings = await self._load_chunk_embeddings(staged)
            staged = self._reranker.rerank(staged, signals.embedding, chunk_embeddings)

        # Step 4: Build injection plan (knapsack)
        plan = self._injector.plan(
            staged=staged,
            token_budget=settings.max_inject_tokens,
            soft_trigger=user_message,
        )

        # Step 5: Mark injected memories
        injected_parent_ids = list({c.parent_id for c in plan.chunks})
        await self._cache.mark_injected(injected_parent_ids)

        # Step 6: Call LLM with enriched context
        llm_response = await self._call_llm(user_message, plan)

        hot_path_ms = (time.monotonic() - t0) * 1000

        log.info(
            "session.hot_path_complete",
            turn=self._turn_index,
            latency_ms=round(hot_path_ms, 1),
            tokens_injected=plan.tokens_used,
            memories_injected=plan.memories_injected,
        )

        # ── POST-RESPONSE (schedule as background tasks) ───────────────────────

        asyncio.create_task(
            self._post_response(
                user_message=user_message,
                llm_response=llm_response,
                signals=signals,
                source=source,
            ),
            name=f"post_response_turn_{self._turn_index}",
        )

        total_ms = (time.monotonic() - t0) * 1000

        return TurnResult(
            session_id=self.session_id,
            turn_index=self._turn_index,
            user_message=user_message,
            llm_response=llm_response,
            injection_plan=plan,
            intent=signals.intent,
            velocity=self._state.current_velocity,
            hit_rate=0.0,                    # updated after feedback runs
            tokens_injected=plan.tokens_used,
            memories_injected=plan.memories_injected,
            latency_ms=total_ms,
            staged_slot_summary=self._cache.slot_summary(),
        )

    async def _post_response(
        self,
        user_message: str,
        llm_response: str,
        signals,
        source: str,
    ) -> None:
        """
        All post-response work runs here asynchronously.
        Errors are logged, never propagated.
        """
        try:
            # 1. Schedule next prefetch (most important — do this first)
            predictions = self._predictor.predict(signals)
            await self._cache.schedule_prefetch(predictions)

            # 2. Feedback: hit/miss detection
            feedback = await self._tracker.evaluate_turn(
                response_text=llm_response,
                intent=signals.intent,
            )

            # 3. Feedback loop: bandits + graph seeds + dataset
            await self._loop.process(feedback, switch_score=self._last_switch_score)

            # 4. Write pipeline: distill + store new memories
            await self._write.process_turn(
                user_message=user_message,
                assistant_message=llm_response,
                source=source,
            )

            log.debug(
                "session.post_response_complete",
                turn=self._turn_index,
                hit_rate=f"{feedback.hit_rate:.2f}",
                hits=feedback.total_hits,
            )

        except Exception as e:
            log.error(
                "session.post_response_error",
                turn=self._turn_index,
                error=str(e),
                exc_info=True,
            )

    async def _call_llm(self, user_message: str, plan: InjectionPlan) -> str:
        """Build prompt with injected context and call LLM."""
        if plan.chunks:
            prompt = f"{plan.context_text}\n\n---\n\nUser: {user_message}"
        else:
            prompt = user_message
        return await self._llm(prompt)

    async def _load_chunk_embeddings(self, staged) -> dict:
        """
        Load chunk embeddings for reranking.
        Returns empty dict if unavailable — reranker handles gracefully.
        """
        try:
            chunk_ids = [c.chunk_id for sm in staged for c in sm.chunks]
            # In production: fetch from Qdrant by IDs
            # For now: return empty (reranker falls back to raw confidence)
            return {}
        except Exception:
            return {}

    def _deduplicate_staged(self, staged) -> list:
        """Remove duplicate StagedMemory objects by parent_id."""
        seen, result = set(), []
        for sm in staged:
            for c in sm.chunks:
                if c.parent_id not in seen:
                    seen.add(c.parent_id)
                    result.append(sm)
                    break
        return result

    async def close(self) -> None:
        """
        Clean shutdown: persist bandit state, cancel pending tasks.
        Called when session ends.
        """
        try:
            snap = self._predictor.bandit_snapshot()
            await self._meta.save_bandit_state(self.session_id, snap)
            log.info("session.closed", session=self.session_id,
                     turns=self._turn_index)
        except Exception as e:
            log.error("session.close_error", error=str(e))

    @property
    def turn_count(self) -> int:
        return self._turn_index

    @property
    def velocity(self) -> float:
        return self._state.current_velocity

