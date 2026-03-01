"""
Outbox Worker for PPM.

Background asyncio task that reads PENDING write_outbox entries from SQLite
and applies them to Qdrant (and Kuzu in the future).
Implements exponential backoff and dead-letter routing on failure.
"""

import asyncio
import json
import logging
from typing import Protocol, Any

logger = logging.getLogger(__name__)

class MetaStoreProtocol(Protocol):
    async def get_pending_outbox(self, limit: int = 50) -> list[Any]: ...
    async def mark_outbox_status(self, outbox_id: str, status: str, attempts: int | None = None, error: str | None = None) -> None: ...

class VectorStoreProtocol(Protocol):
    async def upsert(self, chunk_id: str, parent_id: str, content: str, tokens: int, source_type: str, vector: list[float]): ...
    async def delete(self, memory_id: str): ...


class OutboxWorker:
    """
    Background asyncio task for eventual consistency.
    """
    def __init__(self, meta_store: MetaStoreProtocol, vector_store: VectorStoreProtocol,
                 poll_interval_s: float = 0.1, max_attempts: int = 5, backoff_base_s: float = 2.0):
        self.meta = meta_store
        self.vector = vector_store
        self.poll_interval_s = poll_interval_s
        self.max_attempts = max_attempts
        self.backoff_base_s = backoff_base_s
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self):
        """Start the background poll loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self.run())

    async def stop(self):
        """Stop the background poll loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def run(self):
        """Main poll loop."""
        while self._running:
            try:
                rows = await self.meta.get_pending_outbox(limit=50)
                if not rows:
                    await asyncio.sleep(self.poll_interval_s)
                    continue
                    
                for row in rows:
                    if not self._running:
                        break
                    await self._process(row)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in outbox poll loop: {e}")
                await asyncio.sleep(self.poll_interval_s * 10) # backoff loop on severe error

    async def _process(self, row: Any):
        outbox_id = row["id"]
        operation = row["operation"]
        payload_str = row["payload"]
        attempts = row["attempts"]
        
        try:
            # Mark IN_FLIGHT
            await self.meta.mark_outbox_status(outbox_id, "IN_FLIGHT", attempts=attempts)
            
            payload = json.loads(payload_str)
            
            if operation == "UPSERT_VECTOR":
                await self.vector.upsert(
                    chunk_id=payload["chunk_id"],
                    parent_id=payload["parent_id"],
                    content=payload["content"],
                    tokens=payload["tokens"],
                    source_type=payload.get("source_type", "prose"),
                    vector=payload.get("vector", [])
                )
            elif operation == "DELETE_VECTOR":
                await self.vector.delete(memory_id=payload["memory_id"])
            # edge operations to be implemented in Kuzu graph
            elif operation in ("UPSERT_EDGE", "DELETE_NODE"):
                pass
            else:
                logger.warning(f"Unknown outbox operation: {operation}")
                
            # Success
            await self.meta.mark_outbox_status(outbox_id, "DONE")
            
        except Exception as e:
            attempts += 1
            if attempts >= self.max_attempts:
                # Dead letter
                await self.meta.mark_outbox_status(outbox_id, "DEAD", attempts=attempts, error=str(e))
                logger.error(f"Outbox item {outbox_id} DEAD: {operation} -> {e}")
            else:
                # Setup backoff retry (handled by PENDING status being re-polled, though we need a backoff delay in reality)
                # For Phase 1 we just mark back to PENDING and rely on last_tried filter in a full impl.
                # Since schema is simple, marking PENDING immediately means it retries fast. 
                # In production, query would use `last_tried < now - backoff`
                await self.meta.mark_outbox_status(outbox_id, "PENDING", attempts=attempts)
                logger.warning(f"Outbox item {outbox_id} failed attempt {attempts}: {e}")
                await asyncio.sleep(self.backoff_base_s ** attempts) # simple inline backoff for this worker
