"""
SQLite MetaStore — source of truth for all PPM data.
Qdrant and Kuzu are derived projections rebuilt from this store.

Uses aiosqlite for async, non-blocking I/O.
"""

import aiosqlite
import json
import time
import uuid
from pathlib import Path
from typing import AsyncIterator
from core.types import MemoryID, ChunkID


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    content       TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    source        TEXT,
    source_type   TEXT NOT NULL DEFAULT 'prose',
    token_count   INTEGER NOT NULL DEFAULT 0,
    created_at    INTEGER NOT NULL,
    last_accessed INTEGER,
    access_count  INTEGER NOT NULL DEFAULT 0,
    version       INTEGER NOT NULL DEFAULT 1,
    parent_id     TEXT REFERENCES memories(id),
    deleted_at    INTEGER          -- soft delete
);

CREATE TABLE IF NOT EXISTS chunks (
    id            TEXT PRIMARY KEY,
    parent_id     TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    content       TEXT NOT NULL,
    tokens        INTEGER NOT NULL,
    source_type   TEXT NOT NULL,
    created_at    INTEGER NOT NULL,
    UNIQUE(parent_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS forward_annotations (
    memory_id     TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    context_tag   TEXT NOT NULL,
    weight        REAL NOT NULL DEFAULT 1.0,
    created_at    INTEGER NOT NULL,
    hit_count     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (memory_id, context_tag)
);

CREATE TABLE IF NOT EXISTS graph_edges (
    id            TEXT PRIMARY KEY,
    from_id       TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    to_id         TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    edge_type     TEXT NOT NULL,  -- CALLS|IMPORTS|RELATED_TO|CONFLICTS_WITH|SUMMARIZES
    weight        REAL NOT NULL DEFAULT 1.0,
    created_at    INTEGER NOT NULL
);

-- Outbox for eventual consistency to Qdrant + Kuzu
CREATE TABLE IF NOT EXISTS write_outbox (
    id          TEXT PRIMARY KEY,
    operation   TEXT NOT NULL,  -- UPSERT_VECTOR|DELETE_VECTOR|UPSERT_EDGE|DELETE_NODE
    payload     TEXT NOT NULL,  -- JSON
    status      TEXT NOT NULL DEFAULT 'PENDING',
    attempts    INTEGER NOT NULL DEFAULT 0,
    created_at  INTEGER NOT NULL,
    last_tried  INTEGER
);

CREATE TABLE IF NOT EXISTS dead_letter (
    outbox_id   TEXT PRIMARY KEY,
    error       TEXT,
    failed_at   INTEGER NOT NULL
);

-- Bandit state persistence
CREATE TABLE IF NOT EXISTS bandit_state (
    key         TEXT PRIMARY KEY,  -- "STRATEGY:INTENT"
    alpha       REAL NOT NULL DEFAULT 1.0,
    beta        REAL NOT NULL DEFAULT 1.0,
    updated_at  INTEGER NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_memories_deleted ON memories(deleted_at);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);
CREATE INDEX IF NOT EXISTS idx_fa_tag ON forward_annotations(context_tag, weight DESC);
CREATE INDEX IF NOT EXISTS idx_edges_from ON graph_edges(from_id);
CREATE INDEX IF NOT EXISTS idx_edges_to ON graph_edges(to_id);
CREATE INDEX IF NOT EXISTS idx_outbox_status ON write_outbox(status, created_at);
"""


class MetaStore:
    """
    Async SQLite wrapper. Source of truth for PPM.
    All writes are transactional. Outbox entries created atomically with data writes.
    """

    def __init__(self, db_path: str):
        self._path = db_path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    # ── Memories ──────────────────────────────────────────────────────────────

    async def insert_memory(self, memory: dict) -> MemoryID:
        """Insert memory + chunks + annotations in one transaction."""
        mid = memory.get("id") or str(uuid.uuid4())
        now = int(time.time())
        import hashlib
        h = hashlib.sha256(memory["content"].encode()).hexdigest()

        async with self._db.execute(
            """INSERT OR IGNORE INTO memories
               (id, content, content_hash, source, source_type,
                token_count, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (mid, memory["content"], h, memory.get("source", ""),
             memory.get("source_type", "prose"),
             memory.get("token_count", 0), now)
        ):
            pass

        for chunk in memory.get("chunks", []):
            await self._db.execute(
                """INSERT OR REPLACE INTO chunks
                   (id, parent_id, chunk_index, content, tokens, source_type, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (chunk["id"], mid, chunk["chunk_index"],
                 chunk["content"], chunk["tokens"],
                 chunk.get("source_type", memory.get("source_type", "prose")), now)
            )
            # Enqueue vector upsert for this chunk
            await self._enqueue_outbox(
                "UPSERT_VECTOR",
                {"chunk_id": chunk["id"], "parent_id": mid,
                 "content": chunk["content"], "tokens": chunk["tokens"],
                 "source_type": chunk.get("source_type", "prose")}
            )

        for tag in memory.get("forward_contexts", []):
            await self._db.execute(
                """INSERT OR IGNORE INTO forward_annotations
                   (memory_id, context_tag, created_at) VALUES (?,?,?)""",
                (mid, tag, now)
            )

        for edge in memory.get("graph_edges", []):
            eid = str(uuid.uuid4())
            await self._db.execute(
                """INSERT OR IGNORE INTO graph_edges
                   (id, from_id, to_id, edge_type, weight, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (eid, mid, edge["to_id"],
                 edge.get("type", "RELATED_TO"),
                 edge.get("weight", 1.0), now)
            )
            await self._enqueue_outbox(
                "UPSERT_EDGE",
                {"from_id": mid, "to_id": edge["to_id"],
                 "edge_type": edge.get("type", "RELATED_TO"),
                 "weight": edge.get("weight", 1.0)}
            )

        await self._db.commit()
        return MemoryID(mid)

    async def soft_delete(self, memory_id: MemoryID) -> None:
        now = int(time.time())
        await self._db.execute(
            "UPDATE memories SET deleted_at=? WHERE id=?", (now, memory_id)
        )
        await self._enqueue_outbox("DELETE_VECTOR", {"memory_id": memory_id})
        await self._enqueue_outbox("DELETE_NODE", {"memory_id": memory_id})
        await self._db.commit()

    async def get_recently_written(self, within_seconds: float = 5.0) -> list[str]:
        """Read-your-writes: IDs written in the last N seconds."""
        cutoff = int(time.time() - within_seconds)
        async with self._db.execute(
            "SELECT id FROM memories WHERE created_at >= ? AND deleted_at IS NULL",
            (cutoff,)
        ) as cur:
            return [row[0] async for row in cur]

    # ── Outbox ─────────────────────────────────────────────────────────────────

    async def _enqueue_outbox(self, operation: str, payload: dict) -> None:
        """Must be called within an active transaction."""
        await self._db.execute(
            """INSERT INTO write_outbox (id, operation, payload, created_at)
               VALUES (?,?,?,?)""",
            (str(uuid.uuid4()), operation, json.dumps(payload), int(time.time()))
        )

    async def get_pending_outbox(self, limit: int = 50) -> list[aiosqlite.Row]:
        async with self._db.execute(
            """SELECT * FROM write_outbox
               WHERE status='PENDING'
               ORDER BY created_at ASC LIMIT ?""",
            (limit,)
        ) as cur:
            return await cur.fetchall()

    async def mark_outbox_status(self, outbox_id: str, status: str,
                                  attempts: int | None = None,
                                  error: str | None = None) -> None:
        now = int(time.time())
        if status == "DEAD" and error:
            await self._db.execute(
                "INSERT OR REPLACE INTO dead_letter VALUES (?,?,?)",
                (outbox_id, error, now)
            )
        if attempts is not None:
            await self._db.execute(
                """UPDATE write_outbox
                   SET status=?, attempts=?, last_tried=? WHERE id=?""",
                (status, attempts, now, outbox_id)
            )
        else:
            await self._db.execute(
                "UPDATE write_outbox SET status=?, last_tried=? WHERE id=?",
                (status, now, outbox_id)
            )
        await self._db.commit()

    # ── Forward Annotations ────────────────────────────────────────────────────

    async def search_by_annotation(
        self, context_tags: list[str], limit: int = 20
    ) -> list[dict]:
        """Find memories whose forward annotations match any of the given tags."""
        placeholders = ",".join("?" * len(context_tags))
        async with self._db.execute(
            f"""SELECT m.id, m.content, m.source_type,
                       fa.context_tag, fa.weight
                FROM forward_annotations fa
                JOIN memories m ON fa.memory_id = m.id
                WHERE fa.context_tag IN ({placeholders})
                  AND m.deleted_at IS NULL
                ORDER BY fa.weight DESC, fa.hit_count DESC
                LIMIT ?""",
            (*context_tags, limit)
        ) as cur:
            return [dict(row) async for row in cur]

    async def increment_annotation_hit(
        self, memory_id: MemoryID, context_tag: str
    ) -> None:
        await self._db.execute(
            """UPDATE forward_annotations SET hit_count = hit_count + 1
               WHERE memory_id=? AND context_tag=?""",
            (memory_id, context_tag)
        )
        await self._db.commit()

    # ── Bandit State Sync ──────────────────────────────────────────────────────

    async def save_bandit_state(self, session_id: str, snapshot: dict) -> None:
        import time
        for key, vals in snapshot.items():
            await self._db.execute(
                """INSERT OR REPLACE INTO bandit_state
                   (key, alpha, beta, updated_at) VALUES (?,?,?,?)""",
                (f"{session_id}:{key}", vals["alpha"], vals["beta"], int(time.time()))
            )
        await self._db.commit()

    async def load_bandit_state(self, session_id: str) -> dict | None:
        prefix = f"{session_id}:"
        async with self._db.execute(
            "SELECT key, alpha, beta FROM bandit_state WHERE key LIKE ?",
            (prefix + "%",)
        ) as cur:
            rows = await cur.fetchall()
        if not rows:
            return None
        return {
            row["key"][len(prefix):]: {"alpha": row["alpha"], "beta": row["beta"]}
            for row in rows
        }

