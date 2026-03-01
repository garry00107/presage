"""
ConflictResolver — determines how a new memory relates to existing memories
and decides the appropriate write strategy.

Conflict taxonomy:
  DUPLICATE   → content_hash match or cosine_sim > 0.97
                Action: skip write, increment access_count
  CONFLICT    → cosine_sim ∈ [0.80, 0.97), content materially differs
                Action: write new version, link CONFLICTS_WITH edge, keep both
  EXTENSION   → cosine_sim ∈ [0.55, 0.80)
                Action: write new version, link SUMMARIZES/EXTENDS edge, deprecate old
  NOVEL       → cosine_sim < 0.55
                Action: write as independent memory, no versioning needed

The resolver never silently overwrites. Every conflict is recorded in the
graph so the LLM can reason about contradictions explicitly.
"""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np

from core.types import MemoryID, UnitVector


class ConflictType(str, Enum):
    DUPLICATE = "DUPLICATE"
    CONFLICT  = "CONFLICT"
    EXTENSION = "EXTENSION"
    NOVEL     = "NOVEL"


@dataclass
class ConflictResolution:
    conflict_type: ConflictType
    existing_id: MemoryID | None    # None for NOVEL
    similarity: float
    edge_type: str | None           # graph edge to create
    should_write: bool              # False for DUPLICATE
    should_deprecate: bool          # True for EXTENSION


class ConflictThresholds(NamedTuple):
    duplicate: float = 0.97
    conflict:  float = 0.80
    extension: float = 0.55


class ConflictResolver:
    """
    Compares a new memory embedding against existing candidates
    and returns a ConflictResolution describing the write strategy.

    Designed to be called BEFORE writing to the store, so the write
    layer can make a single informed decision about what to persist.
    """

    def __init__(self, thresholds: ConflictThresholds | None = None):
        self.t = thresholds or ConflictThresholds()

    def resolve(
        self,
        new_embedding: UnitVector,
        candidates: list[tuple[MemoryID, UnitVector, str]],
        # candidates: [(id, embedding, content_hash)]
        new_hash: str = "",
    ) -> ConflictResolution:
        """
        Compare new memory against candidate existing memories.

        Args:
            new_embedding: L2-normalized embedding of new content
            candidates:    [(memory_id, embedding, content_hash)] from store
            new_hash:      SHA-256 of new content (for exact dedup)

        Returns:
            ConflictResolution with the recommended write action.
        """
        if not candidates:
            return ConflictResolution(
                conflict_type=ConflictType.NOVEL,
                existing_id=None, similarity=0.0,
                edge_type=None, should_write=True, should_deprecate=False,
            )

        # Find the most similar existing memory
        best_id, best_sim, best_hash = self._best_match(
            new_embedding, new_hash, candidates
        )

        return self._classify(best_id, best_sim, new_hash, best_hash)

    def _best_match(
        self,
        new_emb: UnitVector,
        new_hash: str,
        candidates: list[tuple[MemoryID, UnitVector, str]],
    ) -> tuple[MemoryID, float, str]:
        best_id, best_sim, best_hash = candidates[0][0], -1.0, ""
        for mem_id, emb, h in candidates:
            # Exact hash match → instant duplicate
            if new_hash and h == new_hash:
                return mem_id, 1.0, h
            sim = float(np.dot(new_emb, emb))  # cosine sim (both unit vectors)
            if sim > best_sim:
                best_id, best_sim, best_hash = mem_id, sim, h
        return best_id, best_sim, best_hash

    def _classify(
        self,
        existing_id: MemoryID,
        sim: float,
        new_hash: str,
        existing_hash: str,
    ) -> ConflictResolution:
        if sim >= self.t.duplicate or (new_hash and new_hash == existing_hash):
            return ConflictResolution(
                conflict_type=ConflictType.DUPLICATE,
                existing_id=existing_id, similarity=sim,
                edge_type=None, should_write=False, should_deprecate=False,
            )

        if sim >= self.t.conflict:
            return ConflictResolution(
                conflict_type=ConflictType.CONFLICT,
                existing_id=existing_id, similarity=sim,
                edge_type="CONFLICTS_WITH",
                should_write=True, should_deprecate=False,
            )

        if sim >= self.t.extension:
            return ConflictResolution(
                conflict_type=ConflictType.EXTENSION,
                existing_id=existing_id, similarity=sim,
                edge_type="EXTENDS",
                should_write=True, should_deprecate=True,
            )

        return ConflictResolution(
            conflict_type=ConflictType.NOVEL,
            existing_id=None, similarity=sim,
            edge_type=None, should_write=True, should_deprecate=False,
        )

