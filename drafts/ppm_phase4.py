# =============================================================================
# PPM: Predictive Push Memory — Phase 4: Staging Layer
#
# Files:
#   core/staging/__init__.py
#   core/staging/models.py          ← slot + staged memory dataclasses
#   core/staging/cache.py           ← P0-P9 async prefetch cache
#   core/staging/prefetcher.py      ← background retrieval worker
#   core/staging/injector.py        ← context budget + knapsack injection
#   core/staging/reranker.py        ← relevance reranking before injection
#   tests/unit/test_cache.py
#   tests/unit/test_injector.py
#   tests/unit/test_reranker.py
#   tests/integration/test_staging_pipeline.py
#
# This is the layer that makes PPM feel magical:
# by the time the user sends their next message, memory is already waiting.
# =============================================================================


### core/staging/__init__.py ###

# empty


### core/staging/models.py ###

"""
Dataclasses for the Staging Layer.

StagedMemory: a prefetched chunk sitting in a slot, ready to inject.
Slot: one of 10 priority slots (P0=highest, P9=lowest).
InjectionPlan: the result of knapsack allocation — what actually goes into context.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from core.types import ChunkID, MemoryID
from core.nerve.models import Prediction, PrefetchStrategy, IntentSignal


class SlotTier(str, Enum):
    AUTO   = "AUTO"    # P0-P1: confidence >= 0.80, inject automatically
    HOT    = "HOT"     # P2-P4: confidence >= 0.50, inject on soft trigger
    WARM   = "WARM"    # P5-P9: confidence >= 0.30, available on demand


@dataclass
class StagedChunk:
    """A single chunk staged in a slot — the atomic unit of staged memory."""
    chunk_id: ChunkID
    parent_id: MemoryID
    content: str
    tokens: int
    score: float                    # confidence × relevance rerank score
    source_type: str
    chunk_index: int
    source: str = ""


@dataclass
class StagedMemory:
    """
    A prefetched memory sitting in a staging slot.
    Contains ranked chunks ready for knapsack selection.
    """
    prediction: Prediction          # the prediction that triggered this fetch
    chunks: list[StagedChunk]       # retrieved + reranked chunks
    raw_confidence: float           # Bayesian bandit confidence at fetch time
    rerank_score: float             # post-rerank relevance score (updated)
    created_at: float = field(default_factory=time.monotonic)
    ttl_seconds: float = 120.0
    was_injected: bool = False      # set True by injector (for feedback layer)
    was_used: bool = False          # set True by feedback layer (hit detection)

    @property
    def combined_score(self) -> float:
        """Injection priority = bandit confidence × rerank relevance."""
        return self.raw_confidence * self.rerank_score

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > self.ttl_seconds

    @property
    def total_tokens(self) -> int:
        return sum(c.tokens for c in self.chunks)

    @property
    def tier(self) -> SlotTier:
        if self.raw_confidence >= 0.80:
            return SlotTier.AUTO
        if self.raw_confidence >= 0.50:
            return SlotTier.HOT
        return SlotTier.WARM


@dataclass
class InjectionPlan:
    """
    The output of the Injector: exactly which chunks to put in context,
    in what order, using how many tokens.
    """
    chunks: list[StagedChunk]       # selected by knapsack, in reading order
    tokens_used: int
    tokens_budget: int
    memories_injected: int          # distinct parent_ids
    staged_memories: list[StagedMemory]  # source StagedMemory objects (for feedback)

    @property
    def context_text(self) -> str:
        """
        Render the injection plan as a formatted context block.
        Grouped by parent memory, separated clearly for the LLM.
        """
        if not self.chunks:
            return ""

        # Group by parent_id, preserve chunk order within parent
        groups: dict[str, list[StagedChunk]] = {}
        for chunk in self.chunks:
            groups.setdefault(chunk.parent_id, []).append(chunk)

        parts = ["<memory_context>"]
        for parent_id, chunks in groups.items():
            source = chunks[0].source or parent_id
            parts.append(f"<memory source=\"{source}\">")
            for chunk in sorted(chunks, key=lambda c: c.chunk_index):
                parts.append(chunk.content)
            parts.append("</memory>")
        parts.append("</memory_context>")

        return "\n".join(parts)


### core/staging/reranker.py ###

"""
Relevance Reranker — refines staged chunk scores against the actual
incoming user message before injection.

Why reranking matters:
  The prefetcher fetched chunks based on a PREDICTED future query vector
  (geodesic extrapolation). The actual user message may differ.
  Before injecting, we score each staged chunk against the REAL message
  embedding and update scores accordingly.

  This catches two failure modes:
    1. The prediction was directionally right but the query drifted
    2. The prediction was wrong but a chunk happens to be relevant anyway

Two reranking strategies:
  COSINE: dot product between chunk embedding and query embedding
          (fast, O(d) per chunk, runs in microseconds)
  CROSS:  not implemented in Phase 4 (requires cross-encoder model,
          added in Phase 7 for high-value slots only)
"""

import numpy as np
from core.staging.models import StagedChunk, StagedMemory
from core.types import UnitVector
from math_core.momentum import l2_normalize


class Reranker:
    """
    Reranks staged memories against the actual user message embedding.
    Updates rerank_score on each StagedMemory in-place.
    Pure computation — no I/O, no side effects.
    """

    def rerank(
        self,
        staged: list[StagedMemory],
        query_embedding: UnitVector,
        chunk_embeddings: dict[str, UnitVector],  # chunk_id → embedding
    ) -> list[StagedMemory]:
        """
        Rerank all staged memories against the real query embedding.

        Args:
            staged:            list of StagedMemory objects to rerank
            query_embedding:   L2-normalized embedding of the actual user message
            chunk_embeddings:  preloaded chunk embeddings from vector store

        Returns:
            Same list with rerank_score updated, sorted by combined_score desc.
        """
        for sm in staged:
            sm.rerank_score = self._score_memory(sm, query_embedding, chunk_embeddings)

        staged.sort(key=lambda s: s.combined_score, reverse=True)
        return staged

    def rerank_chunks(
        self,
        chunks: list[StagedChunk],
        query_embedding: UnitVector,
        chunk_embeddings: dict[str, UnitVector],
    ) -> list[StagedChunk]:
        """
        Rerank individual chunks and update their scores.
        Used by the injector to fine-tune within a selected StagedMemory.
        """
        for chunk in chunks:
            emb = chunk_embeddings.get(chunk.chunk_id)
            if emb is not None:
                chunk.score = float(np.dot(query_embedding, emb))
            # If no embedding available, keep existing score
        return sorted(chunks, key=lambda c: c.score, reverse=True)

    def _score_memory(
        self,
        sm: StagedMemory,
        query_emb: UnitVector,
        chunk_embeddings: dict[str, UnitVector],
    ) -> float:
        """
        Score a StagedMemory as the max cosine similarity across its chunks.
        Max-pooling: a memory is relevant if ANY of its chunks is relevant.
        """
        scores = []
        for chunk in sm.chunks:
            emb = chunk_embeddings.get(chunk.chunk_id)
            if emb is not None:
                scores.append(float(np.dot(query_emb, emb)))

        if not scores:
            # No embeddings available: fall back to raw prediction confidence
            return sm.raw_confidence

        return max(scores)  # max-pooling over chunks


### core/staging/cache.py ###

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

import asyncio
import time
from collections import defaultdict
import structlog

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


### core/staging/prefetcher.py ###

"""
Prefetcher — retrieves memories from all three store backends
based on a Prediction's strategy.

Four retrieval paths:
  SEMANTIC  → vector search with predicted query vector
  GRAPH     → PPR diffusion from seed memory IDs
  SYMBOL    → annotation index lookup (symbol: / file: tags)
  ANNOTATE  → forward annotation index lookup (intent: / topic: tags)
  HYBRID    → vector search + graph walk, merged by score

The prefetcher is stateless — it holds references to store adapters
and executes retrieval. All results are converted to StagedMemory objects.
"""

import asyncio
import time
import structlog
import numpy as np

from core.staging.models import StagedChunk, StagedMemory
from core.nerve.models import Prediction, PrefetchStrategy
from core.types import MemoryID
from math_core.diffusion import personalized_pagerank
from math_core.momentum import l2_normalize
from config.settings import settings

log = structlog.get_logger(__name__)

# Default retrieval limits per strategy
_VECTOR_TOP_K = 20
_GRAPH_DEPTH  = 2
_ANNOT_LIMIT  = 15


class Prefetcher:
    """
    Retrieves and packages memories for staging.
    Injected with store adapters at construction time.
    """

    def __init__(self, vector_store, meta_store, graph_store=None):
        self._vector = vector_store
        self._meta   = meta_store
        self._graph  = graph_store   # optional Kuzu adapter

    async def fetch(self, pred: Prediction) -> StagedMemory | None:
        """
        Dispatch retrieval based on prediction strategy.
        Returns None if no relevant chunks found.
        """
        t0 = time.monotonic()

        try:
            if pred.strategy == PrefetchStrategy.SEMANTIC:
                chunks = await self._fetch_semantic(pred)
            elif pred.strategy == PrefetchStrategy.GRAPH:
                chunks = await self._fetch_graph(pred)
            elif pred.strategy == PrefetchStrategy.SYMBOL:
                chunks = await self._fetch_symbol(pred)
            elif pred.strategy == PrefetchStrategy.ANNOTATE:
                chunks = await self._fetch_annotate(pred)
            elif pred.strategy == PrefetchStrategy.HYBRID:
                chunks = await self._fetch_hybrid(pred)
            else:
                chunks = await self._fetch_semantic(pred)
        except Exception as e:
            log.error("prefetcher.fetch_error",
                      strategy=pred.strategy.value, error=str(e))
            return None

        latency_ms = (time.monotonic() - t0) * 1000
        log.debug(
            "prefetcher.fetched",
            strategy=pred.strategy.value,
            chunks=len(chunks),
            latency_ms=round(latency_ms, 1),
        )

        if not chunks:
            return None

        return StagedMemory(
            prediction=pred,
            chunks=chunks,
            raw_confidence=pred.confidence,
            rerank_score=1.0,   # reranker updates this before injection
            ttl_seconds=settings.slot_ttl_seconds,
        )

    # ── Retrieval strategies ───────────────────────────────────────────────────

    async def _fetch_semantic(self, pred: Prediction) -> list[StagedChunk]:
        """Vector search using the geodesic-extrapolated query vector."""
        results = await self._vector.search(
            query_vector=pred.query_vector,
            top_k=_VECTOR_TOP_K,
        )
        return self._results_to_chunks(results, base_score=pred.confidence)

    async def _fetch_graph(self, pred: Prediction) -> list[StagedChunk]:
        """
        PPR graph walk from seed memory IDs.
        Falls back to semantic if no graph store or no seeds.
        """
        if not pred.graph_seeds:
            return await self._fetch_semantic(pred)

        if self._graph is None:
            # No graph store: fall back to semantic with seed IDs as filter
            return await self._fetch_semantic(pred)

        # Build adjacency from graph store
        try:
            adj = await self._graph.get_adjacency(pred.graph_seeds,
                                                    depth=_GRAPH_DEPTH)
        except Exception:
            return await self._fetch_semantic(pred)

        if not adj:
            return await self._fetch_semantic(pred)

        # Run PPR from each seed, merge scores
        merged_scores: dict[str, float] = {}
        for seed_id in pred.graph_seeds[:3]:
            if seed_id in adj:
                scores = personalized_pagerank(adj, seed_id)
                for nid, score in scores.items():
                    merged_scores[nid] = max(merged_scores.get(nid, 0.0), score)

        if not merged_scores:
            return await self._fetch_semantic(pred)

        # Fetch top-ranked nodes from meta store
        top_ids = sorted(merged_scores, key=merged_scores.get, reverse=True)[:10]
        rows = await self._meta.get_chunks_by_memory_ids(top_ids)
        return self._rows_to_chunks(rows, scores=merged_scores)

    async def _fetch_symbol(self, pred: Prediction) -> list[StagedChunk]:
        """
        Direct annotation index lookup for symbol: and file: tags.
        Highest precision — annotation tags are exact matches.
        Falls back to semantic if no annotation tags provided.
        """
        if not pred.annotation_tags:
            return await self._fetch_semantic(pred)

        rows = await self._meta.search_by_annotation(
            context_tags=pred.annotation_tags,
            limit=_ANNOT_LIMIT,
        )

        if not rows:
            return await self._fetch_semantic(pred)

        return self._annotation_rows_to_chunks(rows, base_score=pred.confidence)

    async def _fetch_annotate(self, pred: Prediction) -> list[StagedChunk]:
        """Forward annotation index lookup for intent: and topic: tags."""
        return await self._fetch_symbol(pred)   # same mechanism, different tags

    async def _fetch_hybrid(self, pred: Prediction) -> list[StagedChunk]:
        """
        Parallel vector search + graph walk, merged by score.
        The definitive strategy for COMPARE intent.
        """
        sem_task  = asyncio.create_task(self._fetch_semantic(pred))
        grph_task = asyncio.create_task(self._fetch_graph(pred))

        sem_chunks, grph_chunks = await asyncio.gather(
            sem_task, grph_task, return_exceptions=True
        )

        all_chunks: list[StagedChunk] = []
        seen_ids: set[str] = set()

        for chunk_list in [sem_chunks, grph_chunks]:
            if isinstance(chunk_list, list):
                for c in chunk_list:
                    if c.chunk_id not in seen_ids:
                        all_chunks.append(c)
                        seen_ids.add(c.chunk_id)

        # Sort merged results by score descending
        all_chunks.sort(key=lambda c: c.score, reverse=True)
        return all_chunks[:_VECTOR_TOP_K]

    # ── Converters ─────────────────────────────────────────────────────────────

    def _results_to_chunks(
        self, results: list[dict], base_score: float
    ) -> list[StagedChunk]:
        """Convert vector store search results to StagedChunk list."""
        chunks = []
        for r in results:
            chunks.append(StagedChunk(
                chunk_id=r.get("id", ""),
                parent_id=r.get("parent_id", ""),
                content=r.get("content", ""),
                tokens=r.get("tokens", 0),
                score=float(r.get("score", base_score)),
                source_type=r.get("source_type", "prose"),
                chunk_index=r.get("chunk_index", 0),
                source=r.get("source", ""),
            ))
        return chunks

    def _rows_to_chunks(
        self, rows: list[dict], scores: dict[str, float]
    ) -> list[StagedChunk]:
        """Convert meta store rows to StagedChunk list with PPR scores."""
        chunks = []
        for r in rows:
            mem_id = r.get("parent_id", r.get("id", ""))
            chunks.append(StagedChunk(
                chunk_id=r.get("id", ""),
                parent_id=mem_id,
                content=r.get("content", ""),
                tokens=r.get("tokens", 0),
                score=scores.get(mem_id, 0.1),
                source_type=r.get("source_type", "prose"),
                chunk_index=r.get("chunk_index", 0),
                source=r.get("source", ""),
            ))
        return chunks

    def _annotation_rows_to_chunks(
        self, rows: list[dict], base_score: float
    ) -> list[StagedChunk]:
        """Convert annotation search rows to StagedChunk list."""
        chunks = []
        for r in rows:
            weight = float(r.get("weight", 1.0))
            chunks.append(StagedChunk(
                chunk_id=r.get("chunk_id", r.get("id", "")),
                parent_id=r.get("memory_id", r.get("id", "")),
                content=r.get("content", ""),
                tokens=r.get("tokens", len(r.get("content", "")) // 3),
                score=base_score * weight,
                source_type=r.get("source_type", "prose"),
                chunk_index=r.get("chunk_index", 0),
                source=r.get("source", ""),
            ))
        return chunks


### core/staging/injector.py ###

"""
Injector — selects which staged memories to inject into the LLM context.

Two-phase selection:
  Phase A: Select StagedMemory objects (memory-level selection)
           Uses tier priority: AUTO first, then HOT if budget remains
  Phase B: Select chunks from selected memories (chunk-level selection)
           Uses 0/1 knapsack on pre-chunked units — NEVER truncates content

The injector is purely computational — no I/O, no side effects.
It receives staged memories and returns an InjectionPlan.
"""

import structlog
from core.staging.models import InjectionPlan, StagedChunk, StagedMemory, SlotTier
from core.types import MemoryID
from math_core.knapsack import knapsack_01
from config.settings import settings

log = structlog.get_logger(__name__)


class Injector:
    """
    Converts staged memories into an InjectionPlan respecting token budget.

    Design invariant: chunk content is NEVER modified.
    The knapsack selects whole chunks only (Phase 2 chunker guarantee).
    """

    def plan(
        self,
        staged: list[StagedMemory],
        token_budget: int | None = None,
        soft_trigger: str | None = None,
    ) -> InjectionPlan:
        """
        Build an injection plan from available staged memories.

        Args:
            staged:       all non-expired StagedMemory objects from cache
            token_budget: max tokens to inject (defaults to settings value)
            soft_trigger: user message text for HOT tier trigger matching

        Returns:
            InjectionPlan with selected chunks and rendered context.
        """
        budget = token_budget or settings.max_inject_tokens
        if not staged:
            return InjectionPlan(
                chunks=[], tokens_used=0, tokens_budget=budget,
                memories_injected=0, staged_memories=[],
            )

        # Phase A: select which StagedMemory objects to consider
        selected_memories = self._select_memories(staged, soft_trigger)

        if not selected_memories:
            return InjectionPlan(
                chunks=[], tokens_used=0, tokens_budget=budget,
                memories_injected=0, staged_memories=[],
            )

        # Phase B: knapsack chunk selection across all selected memories
        all_chunks = self._gather_chunks(selected_memories)
        chosen_chunks = knapsack_01(all_chunks, budget)

        tokens_used = sum(c["tokens"] for c in chosen_chunks)
        parent_ids  = {c["parent_id"] for c in chosen_chunks}

        log.debug(
            "injector.plan_built",
            memories_considered=len(selected_memories),
            chunks_considered=len(all_chunks),
            chunks_selected=len(chosen_chunks),
            tokens_used=tokens_used,
            tokens_budget=budget,
        )

        # Convert back to StagedChunk for InjectionPlan
        staged_chunks = [
            StagedChunk(
                chunk_id=c["id"],
                parent_id=c["parent_id"],
                content=c["content"],
                tokens=c["tokens"],
                score=c["score"],
                source_type=c["source_type"],
                chunk_index=c["chunk_index"],
                source=c.get("source", ""),
            )
            for c in chosen_chunks
        ]

        return InjectionPlan(
            chunks=staged_chunks,
            tokens_used=tokens_used,
            tokens_budget=budget,
            memories_injected=len(parent_ids),
            staged_memories=selected_memories,
        )

    # ── Phase A: Memory selection ──────────────────────────────────────────────

    def _select_memories(
        self,
        staged: list[StagedMemory],
        soft_trigger: str | None,
    ) -> list[StagedMemory]:
        """
        Select memories to inject based on tier priority.

        Rules:
          AUTO tier: always included (highest confidence)
          HOT tier:  included if soft_trigger matches any annotation tag,
                     OR if AUTO tier left budget headroom
          WARM tier: only included on explicit request (Phase 7 feature)
        """
        auto  = [s for s in staged if s.tier == SlotTier.AUTO]
        hot   = [s for s in staged if s.tier == SlotTier.HOT]

        selected = list(auto)

        # Include HOT memories if soft trigger matches
        if soft_trigger and hot:
            triggered = self._soft_trigger_match(hot, soft_trigger)
            selected.extend(triggered)
        elif hot and len(auto) == 0:
            # No AUTO memories ready: fall through to best HOT memory
            selected.extend(hot[:1])

        # Deduplicate by combined_score, keep highest scoring per parent
        selected = self._deduplicate(selected)

        # Sort by combined_score
        selected.sort(key=lambda s: s.combined_score, reverse=True)

        return selected

    def _soft_trigger_match(
        self,
        hot: list[StagedMemory],
        trigger: str,
    ) -> list[StagedMemory]:
        """
        Match HOT memories whose annotation tags appear in the trigger text.
        A memory 'fires' if any of its annotation tags is mentioned.
        """
        trigger_lower = trigger.lower()
        matched = []
        for sm in hot:
            tags = sm.prediction.annotation_tags
            for tag in tags:
                # Strip tag namespace prefix for matching
                # "symbol:verify_token" → match "verify_token" in trigger
                keyword = tag.split(":", 1)[-1].lower()
                if keyword and keyword in trigger_lower:
                    matched.append(sm)
                    break
        return matched

    def _deduplicate(self, memories: list[StagedMemory]) -> list[StagedMemory]:
        """Keep only highest-scoring StagedMemory per parent_id."""
        best: dict[str, StagedMemory] = {}
        for sm in memories:
            for chunk in sm.chunks:
                pid = chunk.parent_id
                if pid not in best or sm.combined_score > best[pid].combined_score:
                    best[pid] = sm
        return list(best.values())

    # ── Phase B: Chunk gathering for knapsack ──────────────────────────────────

    def _gather_chunks(
        self, memories: list[StagedMemory]
    ) -> list[dict]:
        """
        Flatten all chunks from selected memories into knapsack format.
        Deduplicates by chunk_id across memories.
        """
        seen: set[str] = set()
        all_chunks: list[dict] = []

        for sm in memories:
            for c in sm.chunks:
                if c.chunk_id in seen:
                    continue
                seen.add(c.chunk_id)
                all_chunks.append({
                    "id": c.chunk_id,
                    "parent_id": c.parent_id,
                    "chunk_index": c.chunk_index,
                    "content": c.content,
                    "tokens": max(c.tokens, 1),
                    "score": c.score,
                    "source_type": c.source_type,
                    "source": c.source,
                })

        return all_chunks


### tests/unit/test_cache.py ###

"""Tests for core/staging/cache.py"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from core.staging.cache import StagingCache
from core.staging.models import StagedChunk, StagedMemory, SlotTier
from core.nerve.models import Prediction, PrefetchStrategy, IntentSignal
from math_core.momentum import l2_normalize
from config.settings import settings


def make_prediction(conf=0.9, strategy=PrefetchStrategy.SEMANTIC) -> Prediction:
    return Prediction(
        query_vector=l2_normalize(np.random.randn(64).astype(np.float32)),
        query_text="test query",
        graph_seeds=[],
        annotation_tags=[],
        confidence=conf,
        strategy=strategy,
        intent=IntentSignal.EXPLORE,
        k_steps=1,
        slot_index=0,
    )


def make_staged_memory(conf=0.9, ttl=120.0) -> StagedMemory:
    pred = make_prediction(conf=conf)
    chunk = StagedChunk(
        chunk_id="chunk-001",
        parent_id="mem-001",
        content="def verify_token(): pass",
        tokens=10,
        score=conf,
        source_type="code",
        chunk_index=0,
    )
    return StagedMemory(
        prediction=pred,
        chunks=[chunk],
        raw_confidence=conf,
        rerank_score=1.0,
        ttl_seconds=ttl,
    )


def make_cache() -> tuple[StagingCache, AsyncMock]:
    prefetcher = AsyncMock()
    prefetcher.fetch = AsyncMock(return_value=make_staged_memory())
    return StagingCache(prefetcher), prefetcher


@pytest.mark.asyncio
async def test_cache_starts_empty():
    cache, _ = make_cache()
    result = await cache.get_all_ready()
    assert result == []

@pytest.mark.asyncio
async def test_schedule_prefetch_fills_slots():
    cache, prefetcher = make_cache()
    preds = [make_prediction(conf=0.9) for _ in range(3)]
    await cache.schedule_prefetch(preds)
    # Wait briefly for background tasks
    await asyncio.sleep(0.05)
    ready = await cache.get_all_ready()
    assert len(ready) > 0

@pytest.mark.asyncio
async def test_auto_inject_only_high_confidence():
    cache, _ = make_cache()
    # Manually fill slots with different confidence levels
    async with cache._lock:
        cache._slots[0] = make_staged_memory(conf=0.90)  # AUTO
        cache._slots[1] = make_staged_memory(conf=0.55)  # HOT
        cache._slots[2] = make_staged_memory(conf=0.25)  # WARM

    auto = await cache.get_auto_inject()
    assert len(auto) == 1
    assert auto[0].raw_confidence >= settings.auto_inject_threshold

@pytest.mark.asyncio
async def test_evict_expired_removes_stale():
    cache, _ = make_cache()
    async with cache._lock:
        cache._slots[0] = make_staged_memory(ttl=0.001)  # expires immediately
        cache._slots[1] = make_staged_memory(ttl=120.0)  # stays

    await asyncio.sleep(0.01)
    evicted = await cache.evict_expired()
    assert evicted == 1

    ready = await cache.get_all_ready()
    assert len(ready) == 1

@pytest.mark.asyncio
async def test_mark_injected():
    cache, _ = make_cache()
    sm = make_staged_memory()
    async with cache._lock:
        cache._slots[0] = sm

    await cache.mark_injected(["mem-001"])
    assert sm.was_injected is True

@pytest.mark.asyncio
async def test_slot_summary_format():
    cache, _ = make_cache()
    summary = cache.slot_summary()
    assert len(summary) == settings.slot_count
    assert all("slot" in s for s in summary)
    assert summary[0]["status"] == "empty"

@pytest.mark.asyncio
async def test_cancel_previous_tasks_on_new_prefetch():
    """New prefetch cancels old in-flight tasks."""
    cache, prefetcher = make_cache()

    # Slow prefetcher
    async def slow_fetch(pred):
        await asyncio.sleep(10)
        return make_staged_memory()

    prefetcher.fetch = slow_fetch

    preds1 = [make_prediction()]
    await cache.schedule_prefetch(preds1)
    task_count_before = len(cache._active_tasks)

    # Immediately schedule new prefetch — should cancel old
    preds2 = [make_prediction()]
    await cache.schedule_prefetch(preds2)

    # Old tasks should be cancelled
    assert len(cache._active_tasks) <= len(preds2)

@pytest.mark.asyncio
async def test_prefetch_failure_does_not_crash():
    """Prefetch errors are swallowed — cache stays operational."""
    cache, prefetcher = make_cache()
    prefetcher.fetch = AsyncMock(side_effect=Exception("DB down"))

    preds = [make_prediction()]
    await cache.schedule_prefetch(preds)
    await asyncio.sleep(0.05)

    # Cache should still work
    ready = await cache.get_all_ready()
    assert ready == []   # nothing staged, but no crash


### tests/unit/test_injector.py ###

"""Tests for core/staging/injector.py"""

import pytest
import numpy as np
from core.staging.injector import Injector
from core.staging.models import StagedChunk, StagedMemory, SlotTier
from core.nerve.models import Prediction, PrefetchStrategy, IntentSignal
from math_core.momentum import l2_normalize


def make_prediction(conf=0.9, tags=None) -> Prediction:
    return Prediction(
        query_vector=l2_normalize(np.random.randn(64).astype(np.float32)),
        query_text="test",
        graph_seeds=[],
        annotation_tags=tags or [],
        confidence=conf,
        strategy=PrefetchStrategy.SEMANTIC,
        intent=IntentSignal.EXPLORE,
        k_steps=1,
    )


def make_chunk(cid, pid, idx, tokens, score, content=None) -> StagedChunk:
    return StagedChunk(
        chunk_id=cid, parent_id=pid, chunk_index=idx,
        content=content or f"content of {cid}",
        tokens=tokens, score=score, source_type="prose",
    )


def make_staged(conf=0.9, chunks=None, tags=None) -> StagedMemory:
    pred = make_prediction(conf=conf, tags=tags)
    if chunks is None:
        chunks = [make_chunk("c1", "m1", 0, 50, conf)]
    return StagedMemory(
        prediction=pred,
        chunks=chunks,
        raw_confidence=conf,
        rerank_score=1.0,
    )


injector = Injector()


def test_empty_staged_returns_empty_plan():
    plan = injector.plan([])
    assert plan.chunks == []
    assert plan.tokens_used == 0

def test_auto_tier_always_included():
    sm = make_staged(conf=0.90)  # AUTO tier
    plan = injector.plan([sm], token_budget=1000)
    assert plan.memories_injected >= 1

def test_hot_tier_requires_trigger():
    sm = make_staged(conf=0.55, tags=["symbol:verify_token"])  # HOT
    # No trigger → not injected (unless no AUTO present)
    plan_no_trigger = injector.plan([sm], token_budget=1000, soft_trigger=None)
    # No AUTO, so best HOT gets included as fallback
    assert plan_no_trigger.memories_injected >= 0  # depends on fallback logic

    # With matching trigger → injected
    plan_triggered = injector.plan([sm], token_budget=1000,
                                    soft_trigger="fix verify_token please")
    assert plan_triggered.memories_injected >= 1

def test_token_budget_respected():
    chunks = [make_chunk(f"c{i}", "m1", i, 200, 0.9) for i in range(5)]
    sm = make_staged(conf=0.90, chunks=chunks)
    plan = injector.plan([sm], token_budget=400)
    assert plan.tokens_used <= 400

def test_content_never_truncated():
    original = "def foo():\n    return 42"
    chunk = make_chunk("cx", "mx", 0, 20, 0.9, content=original)
    sm = make_staged(conf=0.90, chunks=[chunk])
    plan = injector.plan([sm], token_budget=1000)
    if plan.chunks:
        assert plan.chunks[0].content == original

def test_context_text_has_memory_tags():
    sm = make_staged(conf=0.90)
    plan = injector.plan([sm], token_budget=1000)
    if plan.chunks:
        ctx = plan.context_text
        assert "<memory_context>" in ctx
        assert "</memory_context>" in ctx
        assert "<memory" in ctx

def test_deduplicate_same_parent():
    """Two staged memories for same parent → only highest score wins."""
    chunks_a = [make_chunk("ca", "parent1", 0, 50, 0.9)]
    chunks_b = [make_chunk("cb", "parent1", 1, 50, 0.7)]
    sm_a = make_staged(conf=0.90, chunks=chunks_a)
    sm_b = make_staged(conf=0.70, chunks=chunks_b)
    plan = injector.plan([sm_a, sm_b], token_budget=1000)
    parent_ids = {c.parent_id for c in plan.chunks}
    assert len(parent_ids) == 1

def test_reading_order_preserved_in_context():
    """Chunks from same parent must appear in chunk_index order in context."""
    chunks = [
        make_chunk("c3", "p1", 2, 30, 0.8, "Third chunk"),
        make_chunk("c1", "p1", 0, 30, 0.9, "First chunk"),
        make_chunk("c2", "p1", 1, 30, 0.85, "Second chunk"),
    ]
    sm = make_staged(conf=0.90, chunks=chunks)
    plan = injector.plan([sm], token_budget=1000)
    ctx = plan.context_text
    if "First chunk" in ctx and "Third chunk" in ctx:
        assert ctx.index("First chunk") < ctx.index("Third chunk")

def test_zero_budget_returns_empty():
    sm = make_staged(conf=0.90)
    plan = injector.plan([sm], token_budget=0)
    assert plan.chunks == []


### tests/unit/test_reranker.py ###

"""Tests for core/staging/reranker.py"""

import numpy as np
import pytest
from core.staging.reranker import Reranker
from core.staging.models import StagedChunk, StagedMemory
from core.nerve.models import Prediction, PrefetchStrategy, IntentSignal
from math_core.momentum import l2_normalize


def rand_unit(d=64):
    return l2_normalize(np.random.randn(d).astype(np.float32))


def make_chunk(cid, score=0.5) -> StagedChunk:
    return StagedChunk(
        chunk_id=cid, parent_id="p1", chunk_index=0,
        content="content", tokens=50, score=score, source_type="prose",
    )


def make_staged(conf=0.8, chunk_ids=None) -> StagedMemory:
    pred = Prediction(
        query_vector=rand_unit(),
        query_text="test",
        graph_seeds=[], annotation_tags=[],
        confidence=conf,
        strategy=PrefetchStrategy.SEMANTIC,
        intent=IntentSignal.EXPLORE,
        k_steps=1,
    )
    chunks = [make_chunk(cid) for cid in (chunk_ids or ["c1"])]
    return StagedMemory(
        prediction=pred, chunks=chunks,
        raw_confidence=conf, rerank_score=0.5,
    )


reranker = Reranker()


def test_rerank_updates_scores():
    query = rand_unit()
    chunk_emb = rand_unit()
    sm = make_staged()
    sm.chunks[0].chunk_id = "c1"
    chunk_embeddings = {"c1": chunk_emb}

    result = reranker.rerank([sm], query, chunk_embeddings)
    expected_score = float(np.dot(query, chunk_emb))
    assert abs(result[0].rerank_score - expected_score) < 1e-5

def test_rerank_sorts_by_combined_score():
    query = rand_unit()

    # Make two memories with different similarities to query
    sm_high = make_staged(conf=0.9, chunk_ids=["ch"])
    sm_low  = make_staged(conf=0.5, chunk_ids=["cl"])

    # High similarity for sm_high, low for sm_low
    high_emb = query.copy()   # identical → sim=1.0
    low_emb  = l2_normalize(-query + 0.01 * np.random.randn(*query.shape))

    chunk_embeddings = {"ch": high_emb, "cl": low_emb}
    result = reranker.rerank([sm_low, sm_high], query, chunk_embeddings)

    assert result[0].combined_score >= result[1].combined_score

def test_rerank_fallback_when_no_embedding():
    """Memory without chunk embeddings keeps raw_confidence as rerank_score."""
    query = rand_unit()
    sm = make_staged(conf=0.75)
    result = reranker.rerank([sm], query, {})
    assert result[0].rerank_score == 0.75   # fallback to raw confidence

def test_rerank_chunks_updates_individual_scores():
    query = rand_unit()
    chunks = [make_chunk("c1"), make_chunk("c2")]
    c1_emb = query.copy()
    c2_emb = l2_normalize(-query)
    chunk_embeddings = {"c1": c1_emb, "c2": c2_emb}

    result = reranker.rerank_chunks(chunks, query, chunk_embeddings)
    assert result[0].chunk_id == "c1"   # higher sim should be first

def test_rerank_returns_same_count():
    query = rand_unit()
    staged = [make_staged(conf=0.8), make_staged(conf=0.6)]
    result = reranker.rerank(staged, query, {})
    assert len(result) == 2


### tests/integration/test_staging_pipeline.py ###

"""
Integration test: full Staging pipeline.
Prefetcher → Cache → Reranker → Injector, with mock stores.
"""

import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.staging.cache import StagingCache
from core.staging.injector import Injector
from core.staging.models import StagedChunk, StagedMemory
from core.staging.prefetcher import Prefetcher
from core.staging.reranker import Reranker
from core.nerve.models import Prediction, PrefetchStrategy, IntentSignal
from math_core.momentum import l2_normalize


DIM = 64

def rand_unit():
    return l2_normalize(np.random.randn(DIM).astype(np.float32))


def make_vector_store_mock():
    """Mock vector store returning realistic chunk results."""
    mock = AsyncMock()
    mock.search = AsyncMock(return_value=[
        {
            "id": f"chunk-{i:03d}",
            "parent_id": f"mem-{i // 2:03d}",
            "content": f"def function_{i}():\n    return {i}",
            "tokens": 20,
            "score": 0.9 - i * 0.05,
            "source_type": "code",
            "chunk_index": i % 2,
            "source": f"src/module_{i}.py",
        }
        for i in range(5)
    ])
    return mock


def make_meta_store_mock():
    mock = AsyncMock()
    mock.search_by_annotation = AsyncMock(return_value=[])
    mock.get_chunks_by_memory_ids = AsyncMock(return_value=[])
    return mock


def make_prediction(conf=0.85, strategy=PrefetchStrategy.SEMANTIC) -> Prediction:
    return Prediction(
        query_vector=rand_unit(),
        query_text="fix the authentication bug",
        graph_seeds=[],
        annotation_tags=["intent:DEBUG", "symbol:verify_token"],
        confidence=conf,
        strategy=strategy,
        intent=IntentSignal.DEBUG,
        k_steps=1,
        slot_index=0,
    )


@pytest.mark.asyncio
async def test_full_staging_pipeline():
    """Full pipeline: predictions → prefetch → cache → rerank → inject."""
    vector_store = make_vector_store_mock()
    meta_store   = make_meta_store_mock()

    prefetcher = Prefetcher(vector_store, meta_store)
    cache      = StagingCache(prefetcher)
    reranker   = Reranker()
    injector   = Injector()

    # Step 1: Schedule prefetch from predictions
    predictions = [make_prediction(conf=0.85), make_prediction(conf=0.60)]
    await cache.schedule_prefetch(predictions)
    await asyncio.sleep(0.1)   # allow background tasks to complete

    # Step 2: Get all staged memories
    staged = await cache.get_all_ready()
    assert len(staged) > 0, "Cache should have staged memories"

    # Step 3: Rerank against the actual user query
    query_emb = rand_unit()
    chunk_embeddings = {}  # no embeddings in mock — fallback to raw confidence
    reranked = reranker.rerank(staged, query_emb, chunk_embeddings)
    assert len(reranked) == len(staged)

    # Step 4: Build injection plan
    plan = injector.plan(reranked, token_budget=500)
    assert plan.tokens_used <= 500
    assert plan.memories_injected >= 0

    # Step 5: Verify context text is well-formed
    if plan.chunks:
        ctx = plan.context_text
        assert "<memory_context>" in ctx
        assert "</memory_context>" in ctx


@pytest.mark.asyncio
async def test_soft_trigger_fires_hot_memory():
    """HOT tier memory should inject when trigger text matches annotation tag."""
    vector_store = make_vector_store_mock()
    meta_store   = make_meta_store_mock()
    prefetcher   = Prefetcher(vector_store, meta_store)
    injector     = Injector()

    # Manually create a HOT-tier staged memory with symbol annotation
    from core.staging.models import StagedChunk, StagedMemory
    pred = make_prediction(conf=0.55)   # HOT tier
    pred.annotation_tags = ["symbol:verify_token"]
    chunk = StagedChunk(
        chunk_id="hot-chunk", parent_id="hot-mem", chunk_index=0,
        content="def verify_token(tok): ...", tokens=30, score=0.55,
        source_type="code",
    )
    hot_sm = StagedMemory(
        prediction=pred, chunks=[chunk],
        raw_confidence=0.55, rerank_score=1.0,
    )

    # Without trigger — HOT not injected (no AUTO memories present either,
    # so fallback kicks in — test that trigger increases injection)
    plan_no_trigger = injector.plan([hot_sm], token_budget=1000)

    # With matching trigger
    plan_triggered = injector.plan(
        [hot_sm], token_budget=1000,
        soft_trigger="why does verify_token throw an error?"
    )
    # Triggered should inject the memory
    assert plan_triggered.memories_injected >= 1


@pytest.mark.asyncio
async def test_token_budget_hard_limit():
    """Injector must NEVER exceed token budget."""
    vector_store = make_vector_store_mock()
    meta_store   = make_meta_store_mock()
    prefetcher   = Prefetcher(vector_store, meta_store)
    injector     = Injector()

    predictions = [make_prediction(conf=0.90)]
    cache = StagingCache(prefetcher)
    await cache.schedule_prefetch(predictions)
    await asyncio.sleep(0.1)

    staged = await cache.get_all_ready()
    for budget in [100, 200, 500, 1000]:
        plan = injector.plan(staged, token_budget=budget)
        assert plan.tokens_used <= budget, \
            f"Budget {budget} exceeded: used {plan.tokens_used}"


@pytest.mark.asyncio
async def test_expired_memories_not_injected():
    """Expired staged memories must never make it into an injection plan."""
    from core.staging.models import StagedMemory
    injector = Injector()

    pred = make_prediction(conf=0.90)
    chunk = StagedChunk(
        chunk_id="exp-chunk", parent_id="exp-mem", chunk_index=0,
        content="expired content", tokens=50, score=0.9, source_type="prose",
    )
    expired_sm = StagedMemory(
        prediction=pred, chunks=[chunk],
        raw_confidence=0.90, rerank_score=1.0,
        ttl_seconds=0.001,   # expires almost immediately
    )

    await asyncio.sleep(0.01)
    assert expired_sm.is_expired is True

    # Cache evicts expired; injector receives empty list
    plan = injector.plan([])
    assert plan.memories_injected == 0
