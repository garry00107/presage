# Presage Dogfooding Log

> After every `presage chat` session, write two lines:
> 1. What you were trying to do
> 2. Whether the memory injection felt helpful or irrelevant

---

## Session 1 — First Real Run
**Date:** 2026-04-26
**Goal:** Ask about TrajectoryPredictor k-selection logic, see if the system retrieves its own source code.
**Memory injection verdict:** ❌ Zero memories injected across 2 turns. System completely non-functional for retrieval.

### Bugs Found

**Bug 1: OutboxWorker never started**
- `OutboxWorker` class exists and is fully implemented (`core/store/outbox_worker.py`)
- But it's never instantiated or started in `cmd_serve()`, `SessionFactory`, or the API lifespan
- Result: all 130+ ingested memories sat in SQLite's `write_outbox` queue forever, never propagated to Qdrant
- Vector store was completely empty → search returned 0 results
- **Fix applied:** Added direct Qdrant upsert in `/v1/ingest` endpoint (bypasses outbox)

**Bug 2: Outbox payload missing embedding vectors**
- `meta.py:insert_memory()` enqueues `UPSERT_VECTOR` to outbox but doesn't include the embedding
- `outbox_worker.py:_process()` reads `payload.get("vector", [])` → would get empty list
- Even if outbox worker ran, chunks would be upserted with empty vectors → cosine search broken
- This is a design gap between the ingest path (which has embeddings) and the outbox (which doesn't carry them)

**Bug 3: Slot placement doesn't match tier-based retrieval**
- Cold start prediction: conf=0.5 (HOT tier), placed at `_slots[0]` by confidence sort
- `get_auto_inject()` checks `_slots[:2]` but filters for `tier == AUTO` (conf ≥ 0.80) → misses
- `get_hot()` checks `_slots[2:5]` → slot 0 not in range → misses
- Result: prefetched memory (20 chunks, 19526 tokens) sits in slot 0 but is never retrieved
- **Root cause:** Slot index assigned by descending confidence, but retrieval uses hardcoded slot ranges tied to tiers

### Metrics
| Metric | Turn 1 | Turn 2 |
|---|---|---|
| Intent | EXPLORE | EXPLORE |
| Velocity | 0.0 | 0.0 |
| Memories injected | 0 | 0 |
| Latency | 6111ms | 12377ms |
| Slot 0 state | empty | ready (HOT, 0.5, 19526 tokens) |
| Prefetch chunks | — | 20 chunks fetched but never injected |

### Takeaways
- 3 bugs found in 5 minutes of actual usage
- Architecture is sound on paper but the pipeline has real gaps
- The outbox pattern is fundamentally broken for the ingestion use case
- Slot tier/index mismatch means even working prefetch doesn't reach the LLM
- Llama 3.1 8B via NVIDIA NIM hallucinated about Tesla Autopilot when given no context

---
