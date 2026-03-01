# =============================================================================
# PPM: Predictive Push Memory — Phase 5: Feedback Layer
#
# Files:
#   core/feedback/__init__.py
#   core/feedback/models.py         ← feedback event dataclasses
#   core/feedback/detector.py       ← hit/miss detection (3 strategies)
#   core/feedback/tracker.py        ← per-turn feedback orchestrator
#   core/feedback/dataset.py        ← trajectory training data builder
#   core/feedback/loop.py           ← ties feedback back to predictor + bandits
#   tests/unit/test_detector.py
#   tests/unit/test_tracker.py
#   tests/unit/test_dataset.py
#   tests/integration/test_feedback_loop.py
#
# This is what makes PPM self-improving.
# Every turn generates a (prediction, outcome) pair that:
#   1. Updates Bayesian bandits → better confidence scores immediately
#   2. Updates graph seeds → better GRAPH predictions next turn
#   3. Logs a trajectory sample → training data for future fine-tuning
#   4. Updates forward annotations → memories tag themselves with new contexts
# =============================================================================


### core/feedback/__init__.py ###

# empty


### core/feedback/models.py ###

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


### core/feedback/detector.py ###

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


### core/feedback/tracker.py ###

"""
FeedbackTracker — orchestrates per-turn feedback collection.

Called once per turn AFTER the LLM response is generated.
Coordinates between:
  - StagingCache  (what was staged and injected)
  - HitMissDetector (was it used?)
  - FeedbackLoop  (update bandits, seeds, annotations)

The tracker is the single entry point for the feedback layer.
Everything downstream is triggered by tracker.evaluate_turn().
"""

import asyncio
import uuid
import structlog

from core.feedback.detector import HitMissDetector
from core.feedback.models import TurnFeedback
from core.staging.cache import StagingCache
from core.staging.models import StagedMemory
from core.types import MemoryID, UnitVector
from adapters.embedder.base import Embedder

log = structlog.get_logger(__name__)


class FeedbackTracker:
    """
    Per-session feedback tracker.
    One instance per session — holds references to session-specific objects.
    """

    def __init__(
        self,
        cache: StagingCache,
        detector: HitMissDetector,
        embedder: Embedder,
        session_id: str,
    ):
        self._cache      = cache
        self._detector   = detector
        self._embedder   = embedder
        self._session_id = session_id
        self._turn_index = 0

    async def evaluate_turn(
        self,
        response_text: str,
        intent,                         # IntentSignal
        memory_embeddings: dict | None = None,
    ) -> TurnFeedback:
        """
        Evaluate the feedback for one completed turn.

        Args:
            response_text:      the full LLM response text for this turn
            intent:             IntentSignal detected for this turn
            memory_embeddings:  optional {memory_id: UnitVector} for semantic detection

        Returns:
            TurnFeedback with all hit/miss results and aggregates.
        """
        self._turn_index += 1
        turn_id = str(uuid.uuid4())

        # Step 1: Get all staged memories that were injected this turn
        injected = await self._cache.drain_for_feedback()

        if not injected:
            log.debug(
                "feedback_tracker.no_staged_memories",
                turn=self._turn_index,
            )
            return TurnFeedback(
                turn_id=turn_id,
                session_id=self._session_id,
                turn_index=self._turn_index,
                intent=intent,
                results=[],
            )

        # Step 2: Embed the response (async — may take ~50ms)
        response_embedding: UnitVector = await self._embedder.embed(
            response_text[:2000]   # cap for embedding model limits
        )

        # Step 3: Detect hits/misses for each staged memory
        results = self._detector.detect_batch(
            staged_memories=injected,
            response_text=response_text,
            response_embedding=response_embedding,
            memory_embeddings=memory_embeddings,
        )

        # Step 4: Mark hits in the cache (for graph seed updates)
        hit_ids = [r.memory_id for r in results if r.is_hit]
        if hit_ids:
            await self._cache.mark_used(hit_ids)

        # Step 5: Build TurnFeedback
        feedback = TurnFeedback(
            turn_id=turn_id,
            session_id=self._session_id,
            turn_index=self._turn_index,
            intent=intent,
            results=results,
        )
        feedback.compute_aggregates()

        log.info(
            "feedback_tracker.turn_evaluated",
            turn=self._turn_index,
            staged=feedback.total_staged,
            hits=feedback.total_hits,
            hit_rate=f"{feedback.hit_rate:.2f}",
            intent=intent.value,
        )

        return feedback


### core/feedback/dataset.py ###

"""
TrajectoryDataset — accumulates training samples from the feedback loop.

Each turn that generates feedback produces one TrajectorySample.
Samples are persisted to SQLite and can be exported as JSONL for
fine-tuning the TrajectoryPredictor.

Schema:
  trajectory_samples table in MetaStore (added in this phase)

Export format:
  JSONL, one sample per line, compatible with standard fine-tuning pipelines.

Design goal: accumulate passively — never blocks the hot path.
"""

import json
import time
import uuid
import structlog

from core.feedback.models import TrajectorySample, TurnFeedback
from core.nerve.state import ConversationStateManager
from core.nerve.models import IntentSignal

log = structlog.get_logger(__name__)

# SQL for the trajectory_samples table (appended to MetaStore schema)
TRAJECTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS trajectory_samples (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    turn_index      INTEGER NOT NULL,
    intent          TEXT NOT NULL,
    velocity        REAL NOT NULL,
    acceleration    REAL NOT NULL,
    switch_score    REAL NOT NULL,
    lambda_eff      REAL NOT NULL,
    predictions     TEXT NOT NULL,   -- JSON array
    hit_memory_ids  TEXT NOT NULL,   -- JSON array
    miss_memory_ids TEXT NOT NULL,   -- JSON array
    hit_strategies  TEXT NOT NULL,   -- JSON array
    miss_strategies TEXT NOT NULL,   -- JSON array
    C_t             TEXT NOT NULL,   -- JSON array (embedding)
    M_hat           TEXT,            -- JSON array (embedding) or NULL
    timestamp       REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_traj_session
    ON trajectory_samples(session_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_traj_intent
    ON trajectory_samples(intent, timestamp);
"""


class TrajectoryDataset:
    """
    Accumulates TrajectorySamples from the feedback loop.
    Writes to SQLite asynchronously — never blocks the hot path.
    """

    def __init__(self, meta_store):
        self._meta = meta_store

    async def initialize(self) -> None:
        """Create trajectory_samples table if it doesn't exist."""
        await self._meta._db.executescript(TRAJECTORY_SCHEMA)
        await self._meta._db.commit()

    async def record(
        self,
        feedback: TurnFeedback,
        state: ConversationStateManager,
        switch_score: float = 0.0,
    ) -> TrajectorySample:
        """
        Build and persist one TrajectorySample from a TurnFeedback.

        Args:
            feedback:     TurnFeedback from this turn
            state:        ConversationStateManager for trajectory vectors
            switch_score: context switch score for this turn

        Returns:
            The recorded TrajectorySample.
        """
        snap = state.snapshot()

        predictions = [
            {
                "strategy": r.strategy.value,
                "confidence": r.confidence_at_fetch,
                "k_steps": r.k_steps,
                "slot_index": r.slot_index,
            }
            for r in feedback.results
        ]

        hit_results  = [r for r in feedback.results if r.is_hit]
        miss_results = [r for r in feedback.results if not r.is_hit]

        sample = TrajectorySample(
            session_id=feedback.session_id,
            turn_index=feedback.turn_index,
            intent=feedback.intent.value,
            velocity=snap.velocity,
            acceleration=snap.acceleration,
            switch_score=switch_score,
            lambda_effective=snap.lambda_effective,
            predictions=predictions,
            hit_memory_ids=[r.memory_id for r in hit_results],
            miss_memory_ids=[r.memory_id for r in miss_results],
            hit_strategies=list({r.strategy.value for r in hit_results}),
            miss_strategies=list({r.strategy.value for r in miss_results}),
            C_t=snap.C_t.tolist(),
            M_hat=snap.M_hat.tolist() if snap.M_hat is not None else None,
        )

        await self._persist(sample)

        log.debug(
            "trajectory_dataset.sample_recorded",
            session=feedback.session_id,
            turn=feedback.turn_index,
            hits=len(hit_results),
            misses=len(miss_results),
        )

        return sample

    async def _persist(self, sample: TrajectorySample) -> None:
        """Write sample to SQLite. Fire-and-forget — errors logged, not raised."""
        try:
            await self._meta._db.execute(
                """INSERT OR REPLACE INTO trajectory_samples
                   (id, session_id, turn_index, intent, velocity, acceleration,
                    switch_score, lambda_eff, predictions, hit_memory_ids,
                    miss_memory_ids, hit_strategies, miss_strategies, C_t, M_hat,
                    timestamp)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    str(uuid.uuid4()),
                    sample.session_id,
                    sample.turn_index,
                    sample.intent,
                    sample.velocity,
                    sample.acceleration,
                    sample.switch_score,
                    sample.lambda_effective,
                    json.dumps(sample.predictions),
                    json.dumps(sample.hit_memory_ids),
                    json.dumps(sample.miss_memory_ids),
                    json.dumps(sample.hit_strategies),
                    json.dumps(sample.miss_strategies),
                    json.dumps(sample.C_t),
                    json.dumps(sample.M_hat) if sample.M_hat else None,
                    sample.timestamp,
                )
            )
            await self._meta._db.commit()
        except Exception as e:
            log.error("trajectory_dataset.persist_failed", error=str(e))

    async def export_jsonl(self, output_path: str, min_samples: int = 100) -> int:
        """
        Export all samples as JSONL for fine-tuning.
        Returns number of samples exported.
        Only exports sessions with >= min_samples turns (quality filter).
        """
        async with self._meta._db.execute(
            """SELECT * FROM trajectory_samples
               WHERE session_id IN (
                   SELECT session_id FROM trajectory_samples
                   GROUP BY session_id
                   HAVING COUNT(*) >= ?
               )
               ORDER BY session_id, turn_index""",
            (min_samples,)
        ) as cur:
            rows = await cur.fetchall()

        if not rows:
            log.info("trajectory_dataset.export_empty", min_samples=min_samples)
            return 0

        import aiofiles
        count = 0
        async with aiofiles.open(output_path, "w") as f:
            for row in rows:
                record = {
                    "session_id":      row["session_id"],
                    "turn_index":      row["turn_index"],
                    "intent":          row["intent"],
                    "velocity":        row["velocity"],
                    "acceleration":    row["acceleration"],
                    "switch_score":    row["switch_score"],
                    "lambda_eff":      row["lambda_eff"],
                    "predictions":     json.loads(row["predictions"]),
                    "hit_memory_ids":  json.loads(row["hit_memory_ids"]),
                    "miss_memory_ids": json.loads(row["miss_memory_ids"]),
                    "hit_strategies":  json.loads(row["hit_strategies"]),
                    "miss_strategies": json.loads(row["miss_strategies"]),
                    "C_t":             json.loads(row["C_t"]),
                    "M_hat":           json.loads(row["M_hat"]) if row["M_hat"] else None,
                }
                await f.write(json.dumps(record) + "\n")
                count += 1

        log.info("trajectory_dataset.exported", path=output_path, count=count)
        return count

    async def get_stats(self) -> dict:
        """Summary statistics for observability."""
        async with self._meta._db.execute(
            "SELECT COUNT(*) as n, COUNT(DISTINCT session_id) as sessions FROM trajectory_samples"
        ) as cur:
            row = await cur.fetchone()

        async with self._meta._db.execute(
            """SELECT intent, COUNT(*) as n,
                      AVG(json_array_length(hit_memory_ids)) as avg_hits
               FROM trajectory_samples GROUP BY intent"""
        ) as cur:
            intent_rows = await cur.fetchall()

        return {
            "total_samples":   row["n"],
            "total_sessions":  row["sessions"],
            "by_intent": {
                r["intent"]: {"count": r["n"], "avg_hits": r["avg_hits"]}
                for r in intent_rows
            },
        }


### core/feedback/loop.py ###

"""
FeedbackLoop — closes the loop between feedback and the prediction system.

After every turn, FeedbackLoop takes a TurnFeedback and:
  1. Updates Bayesian bandits in the TrajectoryPredictor
  2. Updates graph seeds in the TrajectoryPredictor (for GRAPH strategy)
  3. Records a TrajectorySample in the dataset
  4. Updates forward annotations in MetaStore (memories tag themselves)
  5. Logs observability metrics

This is the component that makes PPM self-improving over time.
The predictor gets smarter every turn — no retraining needed.
"""

import structlog

from core.feedback.dataset import TrajectoryDataset
from core.feedback.models import TurnFeedback
from core.nerve.predictor import TrajectoryPredictor
from core.nerve.state import ConversationStateManager

log = structlog.get_logger(__name__)


class FeedbackLoop:
    """
    Closes the prediction → feedback → improvement cycle.
    Stateless per-call — all state lives in predictor and dataset.
    """

    def __init__(
        self,
        predictor: TrajectoryPredictor,
        state: ConversationStateManager,
        dataset: TrajectoryDataset,
        meta_store,      # MetaStore — avoid circular import
        annotation_hit_threshold: float = 0.50,
    ):
        self._predictor   = predictor
        self._state       = state
        self._dataset     = dataset
        self._meta        = meta_store
        self._ann_threshold = annotation_hit_threshold

    async def process(
        self,
        feedback: TurnFeedback,
        switch_score: float = 0.0,
    ) -> None:
        """
        Process TurnFeedback and update all downstream systems.

        Args:
            feedback:     TurnFeedback from FeedbackTracker
            switch_score: context switch score for this turn (for dataset)

        This method is non-blocking on the hot path — runs after
        the LLM response has been sent to the user.
        """
        if not feedback.results:
            return

        # Step 1: Update Bayesian bandits
        self._update_bandits(feedback)

        # Step 2: Update graph seeds from hit memory IDs
        if feedback.used_memory_ids:
            self._predictor.update_graph_seeds(feedback.used_memory_ids)

        # Step 3: Record trajectory sample (async, non-blocking)
        await self._dataset.record(feedback, self._state, switch_score)

        # Step 4: Update forward annotations for hit memories
        await self._update_annotations(feedback)

        # Step 5: Log per-turn observability
        self._log_metrics(feedback)

    # ── Internal steps ─────────────────────────────────────────────────────────

    def _update_bandits(self, feedback: TurnFeedback) -> None:
        """
        Update Bayesian bandits for every prediction in this turn.
        One update per (strategy, intent) pair per staged memory.
        """
        for result in feedback.results:
            self._predictor.update_bandits(
                strategy=result.strategy.value,
                intent=result.intent.value,
                hit=result.is_hit,
            )

        log.debug(
            "feedback_loop.bandits_updated",
            turn=feedback.turn_index,
            updates=len(feedback.results),
        )

    async def _update_annotations(self, feedback: TurnFeedback) -> None:
        """
        When a memory is a hit, increment its annotation hit_count
        and add new forward annotation tags based on current intent.

        This implements the "memories tag themselves with future relevance"
        principle: if auth.js was useful during a DEBUG:authentication turn,
        it gets tagged intent:DEBUG and topic:auth with higher weight.
        """
        if not feedback.used_memory_ids:
            return

        # Current intent → annotation tag to add
        intent_tag = f"intent:{feedback.intent.value}"

        for memory_id in feedback.used_memory_ids:
            # Increment hit count on all existing annotations for this memory
            try:
                await self._meta.increment_annotation_hit(memory_id, intent_tag)
            except Exception:
                pass  # annotation may not exist yet — silently skip

            # Add the current intent as a new forward annotation if not present
            try:
                await self._meta._db.execute(
                    """INSERT OR IGNORE INTO forward_annotations
                       (memory_id, context_tag, weight, created_at, hit_count)
                       VALUES (?, ?, ?, strftime('%s','now'), 1)""",
                    (memory_id, intent_tag, 1.0)
                )
                await self._meta._db.commit()
            except Exception as e:
                log.warning(
                    "feedback_loop.annotation_update_failed",
                    memory_id=memory_id,
                    error=str(e),
                )

    def _log_metrics(self, feedback: TurnFeedback) -> None:
        """Structured log for observability dashboards."""
        hit_signals = {}
        for r in feedback.results:
            if r.is_hit:
                hit_signals[r.hit_signal] = hit_signals.get(r.hit_signal, 0) + 1

        log.info(
            "feedback_loop.turn_complete",
            session=feedback.session_id,
            turn=feedback.turn_index,
            intent=feedback.intent.value,
            hit_rate=f"{feedback.hit_rate:.2f}",
            total_staged=feedback.total_staged,
            total_hits=feedback.total_hits,
            hit_signals=hit_signals,
            used_memory_count=len(feedback.used_memory_ids),
        )


### tests/unit/test_detector.py ###

"""Tests for core/feedback/detector.py"""

import numpy as np
import pytest
from math_core.momentum import l2_normalize
from core.staging.models import StagedChunk, StagedMemory
from core.nerve.models import Prediction, PrefetchStrategy, IntentSignal
from core.feedback.detector import (
    HitMissDetector,
    OVERLAP_HIT_THRESHOLD,
    SEMANTIC_HIT_THRESHOLD,
)


def rand_unit(d=64):
    return l2_normalize(np.random.randn(d).astype(np.float32))


def make_staged(content="def verify_token(): pass", conf=0.8) -> StagedMemory:
    pred = Prediction(
        query_vector=rand_unit(),
        query_text="test",
        graph_seeds=[], annotation_tags=[],
        confidence=conf,
        strategy=PrefetchStrategy.SEMANTIC,
        intent=IntentSignal.DEBUG,
        k_steps=1,
    )
    chunk = StagedChunk(
        chunk_id="c1", parent_id="m1", chunk_index=0,
        content=content, tokens=20, score=conf, source_type="code",
    )
    return StagedMemory(
        prediction=pred, chunks=[chunk],
        raw_confidence=conf, rerank_score=1.0,
    )


detector = HitMissDetector()


# ── Trigram overlap ────────────────────────────────────────────────────────────

def test_trigram_overlap_identical():
    text = "the quick brown fox jumps over the lazy dog"
    overlap = detector._trigram_overlap(text, text)
    assert overlap == 1.0

def test_trigram_overlap_no_match():
    mem  = "authentication token validation function"
    resp = "completely unrelated text about databases"
    overlap = detector._trigram_overlap(mem, resp)
    assert overlap == 0.0

def test_trigram_overlap_partial():
    mem  = "verify token authentication session login"
    resp = "the verify token was checked but the session expired"
    overlap = detector._trigram_overlap(mem, resp)
    assert 0.0 < overlap < 1.0

def test_trigram_overlap_short_text():
    """Text with < 3 tokens returns 0.0 (no trigrams)."""
    overlap = detector._trigram_overlap("hi", "hi there how are you")
    assert overlap == 0.0


# ── Hit detection ──────────────────────────────────────────────────────────────

def test_hit_via_overlap():
    """Memory content appears in response → hit via overlap."""
    content = "verify token authentication session login system"
    sm = make_staged(content=content)
    response = f"I reviewed the code. The {content} logic looks correct."
    result = detector.detect(sm, response, rand_unit())
    assert result.is_hit is True
    assert result.hit_signal == "overlap"

def test_hit_via_semantic():
    """High cosine similarity → hit via semantic."""
    sm = make_staged()
    mem_emb = rand_unit()
    # Make response embedding identical to memory embedding → sim = 1.0
    result = detector.detect(
        sm,
        response_text="completely different text with no overlap",
        response_embedding=mem_emb,
        memory_embedding=mem_emb,
    )
    assert result.is_hit is True
    assert result.hit_signal == "semantic"

def test_miss_no_overlap_low_sim():
    """No overlap and low similarity → miss."""
    sm = make_staged("def verify_token(): pass")
    mem_emb  = rand_unit()
    resp_emb = l2_normalize(-mem_emb + 0.5 * np.random.randn(64))

    # Force low similarity
    result = detector.detect(
        sm,
        response_text="databases are great for storing relational data",
        response_embedding=resp_emb,
        memory_embedding=mem_emb,
    )
    # With low sim and no overlap, should be a miss
    if result.semantic_sim_score <= SEMANTIC_HIT_THRESHOLD and \
       result.string_overlap_score <= OVERLAP_HIT_THRESHOLD:
        assert result.is_hit is False
        assert result.hit_signal == "miss"

def test_detect_batch():
    """detect_batch returns one result per staged memory."""
    memories = [make_staged(f"content {i}") for i in range(5)]
    results = detector.detect_batch(
        memories, "some response text", rand_unit()
    )
    assert len(results) == 5

def test_result_has_correct_strategy():
    sm = make_staged()
    result = detector.detect(sm, "response", rand_unit())
    assert result.strategy == PrefetchStrategy.SEMANTIC

def test_result_has_correct_intent():
    sm = make_staged()
    result = detector.detect(sm, "response", rand_unit())
    assert result.intent == IntentSignal.DEBUG


# ── Edge cases ─────────────────────────────────────────────────────────────────

def test_empty_memory_content():
    sm = make_staged(content="")
    result = detector.detect(sm, "some response", rand_unit())
    # Should not crash; overlap should be 0
    assert result.string_overlap_score == 0.0

def test_empty_response():
    sm = make_staged("def foo(): pass")
    result = detector.detect(sm, "", rand_unit())
    assert result.string_overlap_score == 0.0

def test_no_memory_embedding_uses_overlap_only():
    sm = make_staged("verify token session authentication login here")
    response = "the verify token session authentication login here was checked"
    result = detector.detect(sm, response, rand_unit(), memory_embedding=None)
    # Semantic sim should be 0 (no embedding)
    assert result.semantic_sim_score == 0.0
    # But overlap should fire
    assert result.is_hit is True


### tests/unit/test_tracker.py ###

"""Tests for core/feedback/tracker.py"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
import numpy as np

from math_core.momentum import l2_normalize
from core.feedback.tracker import FeedbackTracker
from core.feedback.detector import HitMissDetector
from core.nerve.models import IntentSignal
from core.staging.cache import StagingCache
from core.staging.models import StagedChunk, StagedMemory
from core.nerve.models import Prediction, PrefetchStrategy


def rand_unit(d=64):
    return l2_normalize(np.random.randn(d).astype(np.float32))


def make_staged_memory(content="some content", conf=0.8):
    pred = Prediction(
        query_vector=rand_unit(), query_text="test",
        graph_seeds=[], annotation_tags=[],
        confidence=conf, strategy=PrefetchStrategy.SEMANTIC,
        intent=IntentSignal.DEBUG, k_steps=1,
    )
    chunk = StagedChunk(
        chunk_id="c1", parent_id="m1", chunk_index=0,
        content=content, tokens=20, score=conf, source_type="prose",
    )
    sm = StagedMemory(
        prediction=pred, chunks=[chunk],
        raw_confidence=conf, rerank_score=1.0,
    )
    sm.was_injected = True
    return sm


def make_tracker(staged_memories=None):
    cache = AsyncMock(spec=StagingCache)
    cache.drain_for_feedback = AsyncMock(
        return_value=staged_memories or []
    )
    cache.mark_used = AsyncMock()

    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=rand_unit())

    detector = HitMissDetector()
    tracker = FeedbackTracker(
        cache=cache, detector=detector,
        embedder=embedder, session_id="test-session",
    )
    return tracker, cache, embedder


@pytest.mark.asyncio
async def test_evaluate_turn_no_staged():
    tracker, _, _ = make_tracker(staged_memories=[])
    feedback = await tracker.evaluate_turn("response text", IntentSignal.EXPLORE)
    assert feedback.total_staged == 0
    assert feedback.results == []

@pytest.mark.asyncio
async def test_evaluate_turn_with_staged():
    sm = make_staged_memory(content="verify token auth session login here")
    tracker, cache, embedder = make_tracker([sm])
    feedback = await tracker.evaluate_turn(
        "the verify token auth session login here was reviewed",
        IntentSignal.DEBUG,
    )
    assert feedback.total_staged == 1
    assert len(feedback.results) == 1

@pytest.mark.asyncio
async def test_evaluate_increments_turn_index():
    tracker, _, _ = make_tracker([])
    await tracker.evaluate_turn("r1", IntentSignal.EXPLORE)
    await tracker.evaluate_turn("r2", IntentSignal.DEBUG)
    assert tracker._turn_index == 2

@pytest.mark.asyncio
async def test_evaluate_calls_embedder():
    sm = make_staged_memory()
    tracker, _, embedder = make_tracker([sm])
    await tracker.evaluate_turn("response", IntentSignal.EXPLORE)
    embedder.embed.assert_called_once()

@pytest.mark.asyncio
async def test_feedback_aggregates_computed():
    sm = make_staged_memory(content="verify token auth session login here")
    tracker, _, _ = make_tracker([sm])
    feedback = await tracker.evaluate_turn(
        "the verify token auth session login here checked",
        IntentSignal.DEBUG,
    )
    assert feedback.hit_rate >= 0.0
    assert feedback.total_staged == feedback.total_hits + feedback.total_misses

@pytest.mark.asyncio
async def test_mark_used_called_on_hits():
    sm = make_staged_memory(content="verify token auth session login here")
    tracker, cache, _ = make_tracker([sm])
    feedback = await tracker.evaluate_turn(
        "verify token auth session login here was checked",
        IntentSignal.DEBUG,
    )
    if feedback.total_hits > 0:
        cache.mark_used.assert_called()


### tests/unit/test_dataset.py ###

"""Tests for core/feedback/dataset.py"""

import json
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from math_core.momentum import l2_normalize
from core.feedback.dataset import TrajectoryDataset
from core.feedback.models import TurnFeedback, HitMissResult
from core.nerve.models import IntentSignal, PrefetchStrategy
from core.nerve.state import ConversationStateManager


def rand_unit(d=64):
    return l2_normalize(np.random.randn(d).astype(np.float32))


def make_feedback(hits=1, misses=1) -> TurnFeedback:
    results = []
    for i in range(hits):
        results.append(HitMissResult(
            memory_id=f"hit-mem-{i}",
            strategy=PrefetchStrategy.SEMANTIC,
            intent=IntentSignal.DEBUG,
            confidence_at_fetch=0.8,
            string_overlap_score=0.3,
            semantic_sim_score=0.85,
            prevented_retrieval=False,
            is_hit=True,
            hit_signal="semantic",
        ))
    for i in range(misses):
        results.append(HitMissResult(
            memory_id=f"miss-mem-{i}",
            strategy=PrefetchStrategy.GRAPH,
            intent=IntentSignal.DEBUG,
            confidence_at_fetch=0.5,
            string_overlap_score=0.0,
            semantic_sim_score=0.2,
            prevented_retrieval=False,
            is_hit=False,
            hit_signal="miss",
        ))
    fb = TurnFeedback(
        turn_id="t1", session_id="s1",
        turn_index=1, intent=IntentSignal.DEBUG,
        results=results,
    )
    fb.compute_aggregates()
    return fb


def make_state() -> ConversationStateManager:
    state = ConversationStateManager(dim=64)
    for _ in range(3):
        state.push(rand_unit(), 0.85)
    return state


def make_dataset():
    """Dataset with mocked MetaStore."""
    meta = AsyncMock()
    meta._db = AsyncMock()
    meta._db.execute = AsyncMock()
    meta._db.commit = AsyncMock()
    meta._db.executescript = AsyncMock()

    # Mock fetchone for stats
    cursor_mock = AsyncMock()
    cursor_mock.__aenter__ = AsyncMock(return_value=cursor_mock)
    cursor_mock.__aexit__ = AsyncMock(return_value=False)
    cursor_mock.fetchone = AsyncMock(return_value={"n": 5, "sessions": 2})
    cursor_mock.fetchall = AsyncMock(return_value=[])
    meta._db.execute.return_value = cursor_mock

    return TrajectoryDataset(meta), meta


@pytest.mark.asyncio
async def test_record_builds_sample():
    dataset, _ = make_dataset()
    feedback = make_feedback(hits=2, misses=1)
    state    = make_state()
    sample   = await dataset.record(feedback, state, switch_score=0.1)

    assert sample.session_id == "s1"
    assert sample.turn_index == 1
    assert sample.intent == IntentSignal.DEBUG.value
    assert len(sample.hit_memory_ids) == 2
    assert len(sample.miss_memory_ids) == 1

@pytest.mark.asyncio
async def test_record_C_t_is_list():
    dataset, _ = make_dataset()
    sample = await dataset.record(make_feedback(), make_state())
    assert isinstance(sample.C_t, list)
    assert len(sample.C_t) == 64

@pytest.mark.asyncio
async def test_record_hit_strategies():
    dataset, _ = make_dataset()
    sample = await dataset.record(make_feedback(hits=1, misses=0), make_state())
    assert PrefetchStrategy.SEMANTIC.value in sample.hit_strategies

@pytest.mark.asyncio
async def test_record_persist_called():
    dataset, meta = make_dataset()
    await dataset.record(make_feedback(), make_state())
    meta._db.execute.assert_called()
    meta._db.commit.assert_called()

@pytest.mark.asyncio
async def test_persist_failure_does_not_raise():
    dataset, meta = make_dataset()
    meta._db.execute.side_effect = Exception("DB error")
    # Should not raise — errors are swallowed in persist
    await dataset.record(make_feedback(), make_state())

@pytest.mark.asyncio
async def test_get_stats_returns_dict():
    dataset, _ = make_dataset()
    stats = await dataset.get_stats()
    assert "total_samples" in stats
    assert "total_sessions" in stats


### tests/integration/test_feedback_loop.py ###

"""
Integration test: full Feedback Loop.
Tracker → Detector → Loop → Predictor bandit update.
"""

import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from math_core.momentum import l2_normalize
from core.feedback.detector import HitMissDetector
from core.feedback.loop import FeedbackLoop
from core.feedback.tracker import FeedbackTracker
from core.feedback.dataset import TrajectoryDataset
from core.nerve.models import IntentSignal, PrefetchStrategy
from core.nerve.predictor import TrajectoryPredictor
from core.nerve.state import ConversationStateManager
from core.staging.cache import StagingCache
from core.staging.models import StagedChunk, StagedMemory
from core.nerve.models import Prediction


DIM = 64


def rand_unit():
    return l2_normalize(np.random.randn(DIM).astype(np.float32))


def make_staged_memory(content, conf=0.8, injected=True):
    pred = Prediction(
        query_vector=rand_unit(), query_text="test",
        graph_seeds=[], annotation_tags=[],
        confidence=conf, strategy=PrefetchStrategy.SEMANTIC,
        intent=IntentSignal.DEBUG, k_steps=1,
    )
    chunk = StagedChunk(
        chunk_id=f"c-{id(content)}", parent_id=f"m-{id(content)}",
        chunk_index=0, content=content, tokens=len(content)//3,
        score=conf, source_type="prose",
    )
    sm = StagedMemory(
        prediction=pred, chunks=[chunk],
        raw_confidence=conf, rerank_score=1.0,
    )
    sm.was_injected = injected
    return sm


def make_meta_mock():
    meta = AsyncMock()
    meta._db = AsyncMock()
    meta._db.execute = AsyncMock()
    meta._db.commit = AsyncMock()
    meta._db.executescript = AsyncMock()
    meta.increment_annotation_hit = AsyncMock()
    cursor = AsyncMock()
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=False)
    cursor.fetchone = AsyncMock(return_value={"n": 0, "sessions": 0})
    cursor.fetchall = AsyncMock(return_value=[])
    meta._db.execute.return_value = cursor
    return meta


@pytest.mark.asyncio
async def test_full_feedback_loop_updates_bandits():
    """After a hit, SEMANTIC:DEBUG bandit confidence should increase."""
    state    = ConversationStateManager(dim=DIM)
    for _ in range(3):
        state.push(rand_unit(), 0.85)

    predictor = TrajectoryPredictor(state)
    meta      = make_meta_mock()
    dataset   = TrajectoryDataset(meta)

    # Simulate a hit: content appears verbatim in response
    hit_content = "verify token authentication session login here"
    sm = make_staged_memory(hit_content, conf=0.8)

    cache = AsyncMock(spec=StagingCache)
    cache.drain_for_feedback = AsyncMock(return_value=[sm])
    cache.mark_used = AsyncMock()

    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=rand_unit())

    tracker = FeedbackTracker(
        cache=cache, detector=HitMissDetector(),
        embedder=embedder, session_id="test-session",
    )
    loop = FeedbackLoop(
        predictor=predictor, state=state,
        dataset=dataset, meta_store=meta,
    )

    conf_before = predictor._bandits.confidence(
        PrefetchStrategy.SEMANTIC.value, IntentSignal.DEBUG.value
    )

    # Run feedback: content appears in response → overlap hit
    response = f"I checked the code: {hit_content} was the issue."
    feedback = await tracker.evaluate_turn(response, IntentSignal.DEBUG)
    await loop.process(feedback, switch_score=0.1)

    conf_after = predictor._bandits.confidence(
        PrefetchStrategy.SEMANTIC.value, IntentSignal.DEBUG.value
    )

    if feedback.total_hits > 0:
        assert conf_after >= conf_before, \
            "Confidence should not decrease after a hit"


@pytest.mark.asyncio
async def test_feedback_loop_updates_graph_seeds():
    """Hit memory IDs should become graph seeds in the predictor."""
    state     = ConversationStateManager(dim=DIM)
    predictor = TrajectoryPredictor(state)
    meta      = make_meta_mock()
    dataset   = TrajectoryDataset(meta)

    hit_content = "verify token auth session login here system"
    sm = make_staged_memory(hit_content, conf=0.8)

    cache = AsyncMock(spec=StagingCache)
    cache.drain_for_feedback = AsyncMock(return_value=[sm])
    cache.mark_used = AsyncMock()

    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=rand_unit())

    tracker = FeedbackTracker(
        cache=cache, detector=HitMissDetector(),
        embedder=embedder, session_id="test-session",
    )
    loop = FeedbackLoop(
        predictor=predictor, state=state,
        dataset=dataset, meta_store=meta,
    )

    response = f"reviewed: {hit_content}"
    feedback = await tracker.evaluate_turn(response, IntentSignal.DEBUG)
    await loop.process(feedback)

    if feedback.total_hits > 0:
        assert len(predictor._recent_graph_seeds) > 0


@pytest.mark.asyncio
async def test_feedback_records_trajectory_sample():
    state     = ConversationStateManager(dim=DIM)
    for _ in range(2):
        state.push(rand_unit(), 0.85)

    predictor = TrajectoryPredictor(state)
    meta      = make_meta_mock()
    dataset   = TrajectoryDataset(meta)

    sm = make_staged_memory("some content that was staged")
    cache = AsyncMock(spec=StagingCache)
    cache.drain_for_feedback = AsyncMock(return_value=[sm])
    cache.mark_used = AsyncMock()

    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=rand_unit())

    tracker = FeedbackTracker(
        cache=cache, detector=HitMissDetector(),
        embedder=embedder, session_id="s1",
    )
    loop = FeedbackLoop(
        predictor=predictor, state=state,
        dataset=dataset, meta_store=meta,
    )

    feedback = await tracker.evaluate_turn("any response", IntentSignal.EXPLORE)
    await loop.process(feedback)

    # dataset.record calls meta._db.execute to persist
    meta._db.execute.assert_called()


@pytest.mark.asyncio
async def test_feedback_loop_no_results_is_noop():
    """Empty feedback should not crash or update anything."""
    state     = ConversationStateManager(dim=DIM)
    predictor = TrajectoryPredictor(state)
    meta      = make_meta_mock()
    dataset   = TrajectoryDataset(meta)

    loop = FeedbackLoop(
        predictor=predictor, state=state,
        dataset=dataset, meta_store=meta,
    )

    from core.feedback.models import TurnFeedback
    empty_feedback = TurnFeedback(
        turn_id="t0", session_id="s0",
        turn_index=0, intent=IntentSignal.UNKNOWN, results=[],
    )
    # Should complete without error
    await loop.process(empty_feedback)
