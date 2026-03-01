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

