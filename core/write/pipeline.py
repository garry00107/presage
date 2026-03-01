"""
WritePipeline — orchestrates the full write flow for a conversation turn.

Flow:
  Turn (user + assistant messages)
    → Distiller    → [MemoryCandidates]
    → Embedder     → embeddings per candidate
    → Resolver     → ConflictResolution per candidate
    → Chunker      → chunks per candidate
    → Annotator    → forward annotations
    → MetaStore    → atomic write (data + outbox entries)

Each step is independently testable. The pipeline is the only place
that coordinates across modules — no module imports another module here.
"""

import asyncio
import hashlib
import uuid
from dataclasses import dataclass

import structlog

from adapters.embedder.base import Embedder
from core.types import MemoryID, UnitVector
from core.write.chunker import SemanticChunker
from core.write.conflict import ConflictResolver, ConflictType
from core.write.distiller import MemoryDistiller, MemoryCandidate
from core.write.annotator import ForwardAnnotator

log = structlog.get_logger(__name__)


@dataclass
class WriteResult:
    memory_id: MemoryID | None
    action: str          # 'written' | 'duplicate' | 'skipped' | 'error'
    conflict_type: str
    chunks_written: int
    annotations_written: int


class WritePipeline:
    """
    Orchestrates memory formation for a single conversation turn.
    All steps run asynchronously. Embedding and distillation can run
    in parallel where possible.
    """

    def __init__(
        self,
        distiller: MemoryDistiller,
        embedder: Embedder,
        resolver: ConflictResolver,
        chunker: SemanticChunker,
        annotator: ForwardAnnotator,
        meta_store,    # MetaStore — avoid circular import with string type
        vector_store,  # VectorStore adapter
        top_k_candidates: int = 5,
    ):
        self.distiller   = distiller
        self.embedder    = embedder
        self.resolver    = resolver
        self.chunker     = chunker
        self.annotator   = annotator
        self.meta        = meta_store
        self.vector      = vector_store
        self.top_k       = top_k_candidates

    async def process_turn(
        self,
        user_message: str,
        assistant_message: str,
        source: str = "",
        extra_tags: list[str] | None = None,
    ) -> list[WriteResult]:
        """
        Full write pipeline for one conversation turn.
        Returns one WriteResult per memory candidate extracted.
        """
        # Step 1: Distill candidates from turn
        candidates = await self.distiller.distill(
            user_message, assistant_message, source
        )

        if not candidates:
            log.debug("write_pipeline.no_candidates", source=source)
            return []

        # Step 2: Embed all candidates in parallel
        embeddings: list[UnitVector] = await self.embedder.embed_batch(
            [c.content for c in candidates]
        )

        # Step 3: Process each candidate
        tasks = [
            self._process_candidate(c, emb, extra_tags or [])
            for c, emb in zip(candidates, embeddings)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        write_results = []
        for r in results:
            if isinstance(r, Exception):
                log.error("write_pipeline.candidate_failed", error=str(r))
                write_results.append(WriteResult(
                    memory_id=None, action="error",
                    conflict_type="UNKNOWN", chunks_written=0, annotations_written=0
                ))
            else:
                write_results.append(r)

        return write_results

    async def _process_candidate(
        self,
        candidate: MemoryCandidate,
        embedding: UnitVector,
        extra_tags: list[str],
    ) -> WriteResult:
        content_hash = hashlib.sha256(candidate.content.encode()).hexdigest()

        # Step 3a: Find similar existing memories for conflict check
        similar = await self.vector.search(embedding, top_k=self.top_k)
        candidates_for_resolver = [
            (MemoryID(r["id"]), r["embedding"], r.get("content_hash", ""))
            for r in similar
            if "embedding" in r and r["embedding"] is not None
        ]

        # Step 3b: Conflict resolution
        resolution = self.resolver.resolve(
            embedding, candidates_for_resolver, content_hash
        )

        if not resolution.should_write:
            log.debug("write_pipeline.duplicate_skipped",
                      existing_id=resolution.existing_id,
                      sim=f"{resolution.similarity:.3f}")
            # Update access count on existing
            if resolution.existing_id:
                await self.meta.touch_memory(resolution.existing_id)
            return WriteResult(
                memory_id=resolution.existing_id,
                action="duplicate",
                conflict_type=resolution.conflict_type,
                chunks_written=0, annotations_written=0,
            )

        # Step 3c: Chunk the content
        mid = candidate.id
        raw_chunks = self.chunker.chunk(
            candidate.content, mid, candidate.source_type
        )

        # Step 3d: Embed chunks (batch)
        chunk_texts = [c.content for c in raw_chunks]
        chunk_embeddings = await self.embedder.embed_batch(chunk_texts)

        chunks_dicts = []
        for rc, cemb in zip(raw_chunks, chunk_embeddings):
            d = rc.to_dict()
            d["embedding"] = cemb.tolist()
            chunks_dicts.append(d)

        # Step 3e: Generate forward annotations
        all_tags = candidate.forward_contexts + extra_tags
        annotations = self.annotator.annotate(
            memory_id=mid,
            content=candidate.content,
            source=candidate.source,
            source_type=candidate.source_type,
            extra_tags=all_tags,
        )

        # Step 3f: Build graph edges for conflict/extension
        graph_edges = list(candidate.graph_edges)
        if resolution.edge_type and resolution.existing_id:
            graph_edges.append({
                "to_id": resolution.existing_id,
                "type": resolution.edge_type,
                "weight": resolution.similarity,
            })

        # Step 3g: Atomic write to MetaStore (triggers outbox for Qdrant/Kuzu)
        memory_dict = {
            "id": mid,
            "content": candidate.content,
            "source": candidate.source,
            "source_type": candidate.source_type,
            "token_count": sum(c["tokens"] for c in chunks_dicts),
            "chunks": chunks_dicts,
            "forward_contexts": [a.context_tag for a in annotations],
            "graph_edges": graph_edges,
        }
        await self.meta.insert_memory(memory_dict)

        # Step 3h: Deprecate old memory if this is an extension
        if resolution.should_deprecate and resolution.existing_id:
            await self.meta.soft_delete(resolution.existing_id)
            log.info("write_pipeline.deprecated_old",
                     old_id=resolution.existing_id,
                     new_id=mid)

        log.info(
            "write_pipeline.written",
            memory_id=mid,
            conflict_type=resolution.conflict_type,
            chunks=len(chunks_dicts),
            annotations=len(annotations),
        )

        return WriteResult(
            memory_id=mid,
            action="written",
            conflict_type=resolution.conflict_type,
            chunks_written=len(chunks_dicts),
            annotations_written=len(annotations),
        )

