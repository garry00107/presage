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

