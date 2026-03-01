"""
HitMissDetector — determines whether a prefetched memory was actually useful.

Three independent detection signals, combined by priority:

1. STRING OVERLAP (highest precision)
   Measures what fraction of the memory's key phrases appear verbatim
   in the LLM's response. Uses n-gram overlap (trigrams).
   Threshold: overlap_score > 0.15 → hit

2. SEMANTIC SIMILARITY (recall-oriented)
   Cosine similarity between memory embedding and response embedding.
   Catches cases where the LLM paraphrased the memory content rather
   than copying it verbatim.
   Threshold: sim > 0.75 → hit

3. PREVENTED RETRIEVAL (implicit signal)
   If the LLM's response contains content that closely matches a
   staged (but not injected) memory, the prefetch "worked" by
   preventing a retrieval round-trip.
   Threshold: semantic_sim > 0.80 for un-injected memories

Any single signal firing counts as a hit.
The hit_signal field records which signal fired for observability.
"""

import re
from collections import Counter
import numpy as np

from core.feedback.models import HitMissResult
from core.staging.models import StagedMemory
from core.types import MemoryID, UnitVector
from math_core.momentum import l2_normalize


# Thresholds (tunable via config in Phase 7)
OVERLAP_HIT_THRESHOLD    = 0.15
SEMANTIC_HIT_THRESHOLD   = 0.75
PREVENTED_HIT_THRESHOLD  = 0.80


class HitMissDetector:
    """
    Evaluates whether staged memories were used by the LLM.
    Pure computation — no I/O, no side effects.
    """

    def detect(
        self,
        staged_memory: StagedMemory,
        response_text: str,
        response_embedding: UnitVector,
        memory_embedding: UnitVector | None = None,
    ) -> HitMissResult:
        """
        Evaluate one staged memory against the LLM response.

        Args:
            staged_memory:      the memory that was prefetched
            response_text:      the full LLM response text
            response_embedding: L2-normalized embedding of response
            memory_embedding:   embedding of the memory (if available)

        Returns:
            HitMissResult with is_hit and which signal fired.
        """
        memory_text = self._get_memory_text(staged_memory)
        memory_id   = self._get_memory_id(staged_memory)

        # Signal 1: String overlap (trigram)
        overlap = self._trigram_overlap(memory_text, response_text)

        # Signal 2: Semantic similarity
        sem_sim = 0.0
        if memory_embedding is not None:
            sem_sim = float(np.dot(response_embedding, memory_embedding))

        # Signal 3: Prevented retrieval
        # (same as semantic sim for injected memories — high sim means it was used)
        prevented = (not staged_memory.was_injected) and (sem_sim > PREVENTED_HIT_THRESHOLD)

        # Verdict
        if overlap > OVERLAP_HIT_THRESHOLD:
            is_hit, signal = True, "overlap"
        elif sem_sim > SEMANTIC_HIT_THRESHOLD:
            is_hit, signal = True, "semantic"
        elif prevented:
            is_hit, signal = True, "prevented"
        else:
            is_hit, signal = False, "miss"

        return HitMissResult(
            memory_id=memory_id,
            strategy=staged_memory.prediction.strategy,
            intent=staged_memory.prediction.intent,
            confidence_at_fetch=staged_memory.raw_confidence,
            string_overlap_score=overlap,
            semantic_sim_score=sem_sim,
            prevented_retrieval=prevented,
            is_hit=is_hit,
            hit_signal=signal,
            k_steps=staged_memory.prediction.k_steps,
            slot_index=staged_memory.prediction.slot_index,
        )

    def detect_batch(
        self,
        staged_memories: list[StagedMemory],
        response_text: str,
        response_embedding: UnitVector,
        memory_embeddings: dict[MemoryID, UnitVector] | None = None,
    ) -> list[HitMissResult]:
        """Evaluate all staged memories for one turn."""
        results = []
        embs = memory_embeddings or {}
        for sm in staged_memories:
            mid = self._get_memory_id(sm)
            results.append(self.detect(
                sm, response_text, response_embedding,
                memory_embedding=embs.get(mid),
            ))
        return results

    # ── Detection signals ──────────────────────────────────────────────────────

    def _trigram_overlap(self, memory_text: str, response_text: str) -> float:
        """
        Trigram overlap score: |trigrams(memory) ∩ trigrams(response)| / |trigrams(memory)|

        More robust than exact substring match — handles minor paraphrasing.
        Lowercased, punctuation-stripped for normalization.

        Returns 0.0 if memory has no trigrams (< 3 tokens).
        """
        mem_trigrams  = self._extract_trigrams(memory_text)
        resp_trigrams = self._extract_trigrams(response_text)

        if not mem_trigrams:
            return 0.0

        mem_counts  = Counter(mem_trigrams)
        resp_counts = Counter(resp_trigrams)

        # Intersection size (multiset)
        intersection = sum(
            min(mem_counts[tg], resp_counts[tg])
            for tg in mem_counts
        )
        return intersection / len(mem_trigrams)

    def _extract_trigrams(self, text: str) -> list[tuple[str, str, str]]:
        """Extract word trigrams from text."""
        # Normalize: lowercase, strip punctuation, tokenize
        clean = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = clean.split()
        if len(tokens) < 3:
            return []
        return [(tokens[i], tokens[i+1], tokens[i+2])
                for i in range(len(tokens) - 2)]

    def _get_memory_text(self, sm: StagedMemory) -> str:
        """Concatenate all chunk contents for overlap detection."""
        return " ".join(c.content for c in sm.chunks)

    def _get_memory_id(self, sm: StagedMemory) -> MemoryID:
        """Get the parent memory ID from the first chunk."""
        if sm.chunks:
            return MemoryID(sm.chunks[0].parent_id)
        return MemoryID("unknown")

