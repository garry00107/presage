"""
Shared dataclasses for the Nerve Layer.
Kept in a separate file to avoid circular imports.
"""

from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from core.types import UnitVector, MemoryID


class IntentSignal(str, Enum):
    EXPLORE    = "EXPLORE"      # "what is", "explain", "how does"
    DEBUG      = "DEBUG"        # "error", "fix", "why does"
    NAVIGATE   = "NAVIGATE"     # "where is", "find", "show me"
    COMPARE    = "COMPARE"      # "vs", "difference", "better"
    IMPLEMENT  = "IMPLEMENT"    # "write", "create", "add"
    REFLECT    = "REFLECT"      # "earlier", "we decided", "before"
    UNKNOWN    = "UNKNOWN"


class PrefetchStrategy(str, Enum):
    SEMANTIC = "SEMANTIC"   # pure vector search
    GRAPH    = "GRAPH"      # graph walk from seeds
    SYMBOL   = "SYMBOL"     # exact symbol/file lookup
    HYBRID   = "HYBRID"     # vector + graph combined
    ANNOTATE = "ANNOTATE"   # forward annotation index lookup


@dataclass
class TurnSignals:
    """All signals extracted from one conversation turn."""
    embedding: UnitVector
    intent: IntentSignal
    switch_score: float           # cosine distance from previous turn
    lambda_effective: float       # decay used for this turn's state
    did_reset: bool               # True if context switch triggered reset
    raw_text: str = ""
    extracted_symbols: list[str] = field(default_factory=list)
    extracted_files: list[str] = field(default_factory=list)


@dataclass
class Prediction:
    """A single prefetch prediction with all metadata needed for retrieval."""
    query_vector: UnitVector          # for vector search
    query_text: str                   # for keyword/annotation search
    graph_seeds: list[MemoryID]       # for PPR graph walk
    annotation_tags: list[str]        # for forward annotation lookup
    confidence: float                 # Bayesian bandit confidence
    strategy: PrefetchStrategy
    intent: IntentSignal
    k_steps: int = 1                  # how many turns ahead this predicts
    slot_index: int = 0               # assigned staging slot (0=highest priority)


@dataclass
class ConversationState:
    """Snapshot of the current trajectory state. Serializable for persistence."""
    C_t: UnitVector                   # current state vector
    M_hat: UnitVector | None          # smoothed tangent momentum direction
    M_raw: np.ndarray | None          # raw (unsmoothed) momentum for acceleration
    velocity: float                   # scalar velocity
    acceleration: float               # scalar acceleration (second derivative)
    turn_count: int                   # total turns processed
    lambda_effective: float           # decay used for current state
    last_intent: IntentSignal         # most recent intent signal
    did_reset_last: bool              # whether last turn triggered a reset

