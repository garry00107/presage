"""
Dataclasses for the Feedback Layer.

HitMissResult:   outcome of one staged memory for one turn
TurnFeedback:    aggregated feedback across all staged memories for a turn
TrajectorySample: one training data point for future predictor fine-tuning
"""

from dataclasses import dataclass, field
from typing import Any
import time

from core.nerve.models import IntentSignal, PrefetchStrategy
from core.types import MemoryID


@dataclass
class HitMissResult:
    """
    Outcome for one staged memory in one turn.

    Detection is multi-signal — see detector.py for how each
    signal is computed and combined.
    """
    memory_id: MemoryID
    strategy: PrefetchStrategy
    intent: IntentSignal
    confidence_at_fetch: float

    # Detection signals (each independently measurable)
    string_overlap_score: float     # fraction of content appearing in response
    semantic_sim_score: float       # cosine sim between memory and response
    prevented_retrieval: bool       # True if LLM asked for something we pre-staged

    # Final verdict
    is_hit: bool
    hit_signal: str                 # which signal triggered: 'overlap'|'semantic'|'prevented'|'miss'

    # Metadata
    k_steps: int = 1               # how many turns ahead was this prediction
    slot_index: int = 0


@dataclass
class TurnFeedback:
    """
    All feedback for one conversation turn.
    Produced by FeedbackTracker, consumed by FeedbackLoop.
    """
    turn_id: str
    session_id: str
    turn_index: int
    intent: IntentSignal

    results: list[HitMissResult]

    # Aggregates (computed from results)
    total_staged: int = 0
    total_hits: int = 0
    total_misses: int = 0
    hit_rate: float = 0.0

    # Used memory IDs — fed back to predictor as graph seeds
    used_memory_ids: list[MemoryID] = field(default_factory=list)

    # Timestamp
    created_at: float = field(default_factory=time.monotonic)

    def compute_aggregates(self) -> None:
        self.total_staged = len(self.results)
        self.total_hits   = sum(1 for r in self.results if r.is_hit)
        self.total_misses = self.total_staged - self.total_hits
        self.hit_rate     = self.total_hits / max(self.total_staged, 1)
        self.used_memory_ids = [r.memory_id for r in self.results if r.is_hit]


@dataclass
class TrajectorySample:
    """
    One training data point for the trajectory predictor.

    Accumulates in dataset.py. When enough samples exist,
    this dataset can be used to fine-tune the TrajectoryPredictor's
    heuristic rules into a learned model.

    Schema designed to be JSON-serializable for easy export.
    """
    session_id: str
    turn_index: int
    intent: str                         # IntentSignal.value
    velocity: float
    acceleration: float
    switch_score: float
    lambda_effective: float

    # The predictions that were made
    predictions: list[dict]             # [{strategy, confidence, k_steps}]

    # Ground truth: what was actually needed
    hit_memory_ids: list[str]
    miss_memory_ids: list[str]
    hit_strategies: list[str]           # which strategies produced hits
    miss_strategies: list[str]

    # Context vectors (stored as lists for JSON serialization)
    C_t: list[float]                    # conversation state at this turn
    M_hat: list[float] | None          # momentum direction

    timestamp: float = field(default_factory=time.time)

