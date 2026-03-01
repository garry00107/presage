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

