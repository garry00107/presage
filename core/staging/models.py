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

