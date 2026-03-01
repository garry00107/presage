"""
StagingCache — the 10-slot async prefetch cache.

Slot tiers:
  P0-P1 (AUTO):  confidence >= 0.80 → inject automatically, no trigger needed
  P2-P4 (HOT):   confidence >= 0.50 → inject when soft trigger fires
  P5-P9 (WARM):  confidence >= 0.30 → available instantly on explicit request

Concurrency model:
  - One asyncio.Lock per session (no threading, cooperative)
  - Prefetch tasks run as asyncio background Tasks
  - Cache reads are non-blocking (lock only for slot mutation)
  - TTL eviction runs after every turn (lazy eviction)

The cache is the boundary between the Nerve Layer (prediction)
and the retrieval backends (vector store, graph, annotation index).
"""

from __future__ import annotations
import asyncio
import time
from collections import defaultdict
from typing import TYPE_CHECKING
import structlog

if TYPE_CHECKING:
    from core.staging.prefetcher import Prefetcher

from core.staging.models import StagedMemory, SlotTier
from core.nerve.models import Prediction
from core.types import MemoryID
from config.settings import settings

log = structlog.get_logger(__name__)


class StagingCache:
    """
    Priority staging cache with async background prefetch.

    One instance per session. The prefetcher fills slots asynchronously
    while the LLM generates its response — by next turn, slots are ready.
    """

    def __init__(self, prefetcher):
        # Avoid circular import — prefetcher injected at runtime
        self._prefetcher = prefetcher
        self._slots: list[StagedMemory | None] = [None] * settings.slot_count
        self._lock = asyncio.Lock()
        self._active_tasks: set[asyncio.Task] = set()

    # ── Public API ─────────────────────────────────────────────────────────────

    async def schedule_prefetch(self, predictions: list[Prediction]) -> None:
        """
        Schedule async prefetch for a ranked list of predictions.
        Returns immediately — prefetch runs in background.
        Called by the session manager after every turn.
        """
        # Cancel stale prefetch tasks before starting new ones
        await self._cancel_active_tasks()

        for i, pred in enumerate(predictions[:settings.slot_count]):
            task = asyncio.create_task(
                self._prefetch_slot(pred, slot_index=i),
                name=f"prefetch_slot_{i}",
            )
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)

        log.debug(
            "staging_cache.prefetch_scheduled",
            predictions=len(predictions),
            tasks=len(self._active_tasks),
        )

    async def get_auto_inject(self) -> list[StagedMemory]:
        """
        Return all AUTO-tier (P0-P1) staged memories that are ready.
        Called synchronously on the hot path before LLM invocation.
        Non-blocking — returns whatever is ready NOW.
        """
        async with self._lock:
            return [
                sm for sm in self._slots[:2]  # P0-P1 only
                if sm is not None
                and not sm.is_expired
                and sm.tier == SlotTier.AUTO
            ]

    async def get_hot(self, trigger_text: str = "") -> list[StagedMemory]:
        """
        Return HOT-tier (P2-P4) staged memories.
        Used when a soft trigger fires (e.g., user references a symbol
        that matches a staged prediction's annotation tags).
        """
        async with self._lock:
            return [
                sm for sm in self._slots[2:5]  # P2-P4
                if sm is not None and not sm.is_expired
            ]

    async def get_all_ready(self) -> list[StagedMemory]:
        """Return all non-expired staged memories across all slots."""
        async with self._lock:
            return [
                sm for sm in self._slots
                if sm is not None and not sm.is_expired
            ]

    async def evict_expired(self) -> int:
        """
        Remove expired slots. Called after every turn.
        Returns number of slots evicted.
        """
        evicted = 0
        async with self._lock:
            for i, sm in enumerate(self._slots):
                if sm is not None and sm.is_expired:
                    self._slots[i] = None
                    evicted += 1
        if evicted:
            log.debug("staging_cache.evicted", count=evicted)
        return evicted

    async def mark_injected(self, memory_ids: list[MemoryID]) -> None:
        """Mark staged memories as injected (for feedback tracking)."""
        async with self._lock:
            for sm in self._slots:
                if sm is not None:
                    pid = sm.prediction.graph_seeds[0] \
                        if sm.prediction.graph_seeds else None
                    # Mark if any chunk's parent_id is in the injected set
                    for chunk in sm.chunks:
                        if chunk.parent_id in memory_ids:
                            sm.was_injected = True
                            break

    async def mark_used(self, memory_ids: list[MemoryID]) -> None:
        """Called by feedback layer: these memories were actually used by LLM."""
        async with self._lock:
            for sm in self._slots:
                if sm is not None:
                    for chunk in sm.chunks:
                        if chunk.parent_id in memory_ids:
                            sm.was_used = True
                            break

    async def drain_for_feedback(self) -> list[StagedMemory]:
        """
        Return all staged memories that were injected this turn,
        then clear was_injected flag. Used by feedback layer.
        """
        async with self._lock:
            injected = [sm for sm in self._slots
                        if sm is not None and sm.was_injected]
            for sm in injected:
                sm.was_injected = False  # reset for next turn
            return injected

    def slot_summary(self) -> list[dict]:
        """Debug summary of current slot state."""
        summary = []
        for i, sm in enumerate(self._slots):
            if sm is None:
                summary.append({"slot": i, "status": "empty"})
            elif sm.is_expired:
                summary.append({"slot": i, "status": "expired"})
            else:
                summary.append({
                    "slot": i,
                    "status": "ready",
                    "tier": sm.tier.value,
                    "confidence": round(sm.raw_confidence, 3),
                    "rerank_score": round(sm.rerank_score, 3),
                    "tokens": sm.total_tokens,
                    "strategy": sm.prediction.strategy.value,
                })
        return summary

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _prefetch_slot(self, pred: Prediction, slot_index: int) -> None:
        """
        Background task: fetch a prediction and fill a slot.
        Any exception is caught and logged — never propagates to caller.
        """
        try:
            staged = await self._prefetcher.fetch(pred)
            if staged is not None:
                async with self._lock:
                    self._slots[slot_index] = staged
                log.debug(
                    "staging_cache.slot_filled",
                    slot=slot_index,
                    strategy=pred.strategy.value,
                    confidence=round(pred.confidence, 3),
                    chunks=len(staged.chunks),
                    tokens=staged.total_tokens,
                )
        except asyncio.CancelledError:
            pass  # expected on turn boundary
        except Exception as e:
            log.error(
                "staging_cache.prefetch_failed",
                slot=slot_index,
                strategy=pred.strategy.value,
                error=str(e),
            )

    async def _cancel_active_tasks(self) -> None:
        """Cancel all in-flight prefetch tasks from previous turn."""
        if not self._active_tasks:
            return
        for task in list(self._active_tasks):
            task.cancel()
        # Wait briefly for cancellation to propagate
        await asyncio.gather(*self._active_tasks, return_exceptions=True)
        self._active_tasks.clear()

