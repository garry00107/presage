# PPM Refinement Addendum — v2
## Addressing 4 Critical Architectural Issues

---

## REFINEMENT 1 — Chunk-Aware 0/1 Knapsack (Replace Fractional)

### The Problem
Raw string truncation (`content[:n]`) cuts through code mid-function, mid-bracket, mid-sentence. An LLM receiving a truncated JSON object or half a Python function will hallucinate the missing structure. This is catastrophic for a code-focused memory system.

### Why 0/1 is Fine Here
The fractional knapsack's O(N log N) advantage over 0/1 DP's O(N·W) only matters when N and W are large. Our staging cache has at most 10 slots (N=10) and the token budget W is bounded (~4096 max injection budget). The DP table is 10×4096 = 40,960 cells — computed in microseconds. The complexity argument for fractional knapsack does not apply here.

### The Fix — Chunk-Aware 0/1 Knapsack

Every memory is pre-split into **semantic chunks** at write time — not at injection time. Chunks are split at natural boundaries:

- **Code:** AST node boundaries (function def, class def, block end)
- **Prose:** Sentence boundaries (spaCy sentencizer or regex `[.!?]\s`)
- **JSON/YAML:** Top-level key boundaries
- **Markdown:** Header boundaries

Each chunk is stored as an independent unit in the vector store with a `parent_memory_id` and `chunk_index`. The knapsack operates on chunks, not raw content.

```python
# math/knapsack.py

def knapsack_01(chunks: list[dict], budget: int) -> list[dict]:
    """
    True 0/1 knapsack via bottom-up DP.
    Items are pre-chunked semantic units — never truncated.
    
    chunks: [{'id': str, 'tokens': int, 'score': float, 'content': str, 
               'parent_id': str, 'chunk_index': int}]
    budget: int (max tokens to inject)
    
    Returns selected whole chunks, sorted by chunk_index within each parent
    to preserve reading order.
    
    Time:  O(N * W) — negligible for N<=50, W<=4096
    Space: O(N * W) — ~800KB worst case, acceptable
    """
    N = len(chunks)
    # DP table: dp[i][w] = max score using first i items with budget w
    dp = [[0.0] * (budget + 1) for _ in range(N + 1)]
    
    for i, chunk in enumerate(chunks, 1):
        w_i = chunk['tokens']
        v_i = chunk['score']
        for w in range(budget + 1):
            # don't take item i
            dp[i][w] = dp[i-1][w]
            # take item i if it fits
            if w >= w_i:
                dp[i][w] = max(dp[i][w], dp[i-1][w - w_i] + v_i)
    
    # Backtrack to find selected items
    selected_ids = set()
    w = budget
    for i in range(N, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected_ids.add(chunks[i-1]['id'])
            w -= chunks[i-1]['tokens']
    
    selected = [c for c in chunks if c['id'] in selected_ids]
    
    # Restore reading order: group by parent, sort by chunk_index
    from itertools import groupby
    selected.sort(key=lambda c: (c['parent_id'], c['chunk_index']))
    return selected


# core/write/chunker.py

class SemanticChunker:
    """
    Splits a memory into natural-boundary chunks at write time.
    Chunk boundaries depend on source_type.
    """
    def chunk(self, memory: Memory) -> list[Chunk]:
        if memory.source_type == 'code':
            return self._chunk_code(memory)
        elif memory.source_type in ('json', 'yaml'):
            return self._chunk_structured(memory)
        else:
            return self._chunk_prose(memory)
    
    def _chunk_code(self, memory: Memory) -> list[Chunk]:
        """Uses ast.parse to split at top-level node boundaries."""
        import ast
        tree = ast.parse(memory.content)
        chunks = []
        lines = memory.content.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = node.end_lineno
                content = '\n'.join(lines[start:end])
                chunks.append(Chunk(content=content, chunk_index=len(chunks),
                                    parent_id=memory.id))
        return chunks or [Chunk(content=memory.content, chunk_index=0, 
                                parent_id=memory.id)]
    
    def _chunk_prose(self, memory: Memory) -> list[Chunk]:
        """Splits at sentence boundaries."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', memory.content)
        # Group into ~200 token chunks
        chunks, current, idx = [], [], 0
        for s in sentences:
            current.append(s)
            if sum(len(x.split()) for x in current) > 150:
                chunks.append(Chunk(content=' '.join(current),
                                    chunk_index=idx, parent_id=memory.id))
                current, idx = [], idx + 1
        if current:
            chunks.append(Chunk(content=' '.join(current),
                                chunk_index=idx, parent_id=memory.id))
        return chunks
```

**Schema change:** Qdrant stores chunks, not raw memories. Each chunk point has payload `{parent_id, chunk_index, source_type, tokens}`. SQLite `memories` table gains a `chunks` child table.

---

## REFINEMENT 2 — Spherical Geometry for Embedding Space

### The Problem
OpenAI embeddings live on a unit hypersphere S^(d-1) ⊂ ℝ^d (L2-normalized). When you compute:

```
Ĉ_{t+k} = C_t + k·M_t
```

The result vector is **inside the sphere**, not on it. Qdrant's cosine similarity search computes:

```
sim(q, x) = (q · x) / (||q|| · ||x||)
```

If `||q|| ≠ 1`, this is fine for ranking (cosine similarity is invariant to query magnitude), BUT the linear extrapolation itself is geometrically wrong — you are moving in Euclidean space while the meaningful geometry is spherical.

### The Correct Fix — Spherical Linear Interpolation (SLERP)

The proper interpolation between two points on a hypersphere is **SLERP**:

```
SLERP(v₀, v₁, t) = sin((1-t)·Ω)/sin(Ω) · v₀ + sin(t·Ω)/sin(Ω) · v₁

where Ω = arccos(v₀ · v₁)   (angle between vectors, both unit-normalized)
```

For momentum, we don't interpolate between two known points — we **extrapolate** beyond `C_t` in the direction of `M_t`. The correct formulation is:

```
# 1. Project momentum onto the tangent plane at C_t
#    (remove component parallel to C_t — that component only changes magnitude, not direction)
M_tangent = M_t - (M_t · C_t) · C_t

# 2. Normalize tangent momentum to get a unit direction on the sphere
M_hat = M_tangent / ||M_tangent||

# 3. SLERP extrapolation: move angle θ along the geodesic from C_t in direction M_hat
θ = velocity_scalar * step_size   # tunable
Ĉ_{t+k} = cos(θ) · C_t + sin(θ) · M_hat
```

This is **geodesic extrapolation** — moving along the great circle of the hypersphere rather than punching through its interior.

```python
# math/momentum.py — UPDATED

import numpy as np

def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v

def conversation_state(turn_embeddings: list[np.ndarray], 
                        decay: float = 0.85) -> np.ndarray:
    """Exponentially decayed weighted sum, L2-normalized onto unit sphere."""
    weights = np.array([decay ** i for i in range(len(turn_embeddings))][::-1])
    weights /= weights.sum()
    result = weights @ np.stack(turn_embeddings)
    return l2_normalize(result)  # ← project back onto sphere

def momentum_tangent(C_t: np.ndarray, C_prev: np.ndarray,
                     M_prev: np.ndarray | None, beta: float = 0.9
                     ) -> tuple[np.ndarray, float]:
    """
    Computes smoothed momentum projected onto the tangent plane at C_t.
    Both C_t and C_prev must be L2-normalized.
    
    Returns:
        M_tangent: unit tangent vector on sphere at C_t
        velocity:  scalar magnitude (angle moved per turn, in radians)
    """
    # Raw momentum in embedding space
    raw = C_t - C_prev
    
    # Smooth (like Adam m_hat)
    smoothed = beta * (M_prev if M_prev is not None else raw) + (1 - beta) * raw
    
    # Project onto tangent plane: remove component parallel to C_t
    tangent = smoothed - np.dot(smoothed, C_t) * C_t
    
    velocity = float(np.linalg.norm(tangent))
    M_hat = tangent / velocity if velocity > 1e-10 else tangent
    
    return M_hat, velocity

def predict_future_state(C_t: np.ndarray, M_hat: np.ndarray,
                          velocity: float, k: int = 1,
                          step_size: float = 0.3) -> np.ndarray:
    """
    Geodesic extrapolation on the unit hypersphere.
    Moves angle θ = velocity * k * step_size along the great circle
    defined by C_t and M_hat.
    
    Result is guaranteed to be on the unit sphere — safe to use
    directly as a cosine similarity query vector.
    """
    theta = velocity * k * step_size
    # SLERP-style extrapolation: result stays on sphere by construction
    predicted = np.cos(theta) * C_t + np.sin(theta) * M_hat
    # Re-normalize defensively (floating point drift)
    return l2_normalize(predicted)
```

**Invariant enforced throughout:** Every vector that exits the `momentum.py` module is L2-normalized. Type aliases make this explicit:

```python
# core/types.py
from typing import NewType
import numpy as np

UnitVector = NewType('UnitVector', np.ndarray)  
# contract: ||v|| == 1.0 (±1e-6)
# enforced by l2_normalize() at module boundaries
```

---

## REFINEMENT 3 — Dynamic Decay & Bounded State Window

### The Problem
With a large lookback N and fixed λ, the weighted sum `C_t = Σ λ^i · embed(turn_i)` converges toward the centroid of the turn embedding distribution — the "average topic" rather than the current topic. This is "semantic muddying."

Additionally, a fixed λ doesn't react to **context switches** — if a user pivots from debugging auth to asking about database migrations, the momentum vector should reset, not smoothly drift.

### The Fix — Adaptive Decay with Context-Switch Detection

Two components:

**1. Hard-bounded window:** Cap N at 6. Beyond 6 turns back, the marginal information is negligible and muddying dominates.

**2. Context-switch detection:** Measure the cosine distance between adjacent turns. If it exceeds a threshold (a sharp semantic jump), trigger a **decay reset**.

```
context_switch_score = 1 - cosine_sim(embed(turn_t), embed(turn_{t-1}))

if context_switch_score > θ_switch:
    # Hard reset: discard history, start fresh from turn_t
    state_history.clear()
    momentum = zeros(d)
    λ_effective = 0.99   # slow decay — we're in a new topic, stay focused
else:
    # Gradual: decay factor modulated by velocity
    # High velocity (fast topic drift) → lower λ (forget faster)
    # Low velocity (deep dive) → higher λ (remember more)
    λ_effective = λ_base - α * velocity   # α tunable, e.g. 0.1
    λ_effective = clip(λ_effective, 0.6, 0.95)
```

```python
# math/entropy.py

def context_switch_score(embed_prev: np.ndarray, embed_curr: np.ndarray) -> float:
    """
    Cosine distance between adjacent turn embeddings.
    0.0 = identical topic, 1.0 = orthogonal (complete switch), >1.0 = opposite direction.
    """
    return float(1.0 - np.dot(embed_prev, embed_curr))  # both unit vectors

def adaptive_decay(lambda_base: float, velocity: float,
                   switch_score: float,
                   switch_threshold: float = 0.4,
                   alpha: float = 0.1) -> tuple[float, bool]:
    """
    Returns (effective_lambda, did_reset).
    did_reset=True signals caller to clear state history.
    """
    if switch_score > switch_threshold:
        return 0.99, True   # reset — fresh start with slow forgetting
    lam = lambda_base - alpha * velocity
    return float(np.clip(lam, 0.6, 0.95)), False
```

**Updated ConversationObserver:**

```python
async def on_turn_complete(self, turn: Turn) -> TurnSignals:
    embed = await self.embedder.embed(turn.text)
    
    switch_score = context_switch_score(self.last_embed, embed) \
                   if self.last_embed is not None else 0.0
    lam, did_reset = adaptive_decay(self.lambda_base, self.velocity, switch_score)
    
    if did_reset:
        self.state_history.clear()
        self.predictor.reset_momentum()
    
    self.state_history.append(embed)
    if len(self.state_history) > 6:   # hard cap
        self.state_history.popleft()
    
    self.last_embed = embed
    return TurnSignals(embedding=embed, switch_score=switch_score,
                       lambda_effective=lam, did_reset=did_reset)
```

**Observability:** Log `switch_score`, `lambda_effective`, and `did_reset` on every turn. This gives you a clear diagnostic signal for tuning thresholds per use case (coding vs. general chat will have different typical switch scores).

---

## REFINEMENT 4 — Cross-Store Transactional Integrity

### The Problem
A write to `MemoryStore` touches three independent databases (SQLite, Qdrant, Kuzu). If Kuzu succeeds but Qdrant fails, the stores are inconsistent. There is no native distributed transaction across these systems. An eviction or conflict resolution that partially succeeds will corrupt the referential integrity.

### The Fix — Outbox Pattern + Compensating Transactions

This is a well-established distributed systems pattern. The key insight: **SQLite is the source of truth**. Qdrant and Kuzu are *derived projections*. They can always be rebuilt from SQLite. Therefore:

**1. Write-Ahead Outbox in SQLite**

Every intended write is first logged to an `outbox` table. A background worker reads the outbox and applies writes to Qdrant and Kuzu. On failure, it retries with exponential backoff. On permanent failure, it writes to a `dead_letter` table for manual inspection.

```sql
-- SQLite outbox table
CREATE TABLE write_outbox (
    id          TEXT PRIMARY KEY,
    operation   TEXT NOT NULL,   -- 'UPSERT_VECTOR' | 'DELETE_VECTOR' | 
                                 -- 'UPSERT_EDGE' | 'DELETE_NODE'
    payload     TEXT NOT NULL,   -- JSON
    status      TEXT DEFAULT 'PENDING',  -- PENDING | IN_FLIGHT | DONE | DEAD
    attempts    INTEGER DEFAULT 0,
    created_at  INTEGER,
    last_tried  INTEGER
);

CREATE TABLE dead_letter (
    outbox_id   TEXT PRIMARY KEY,
    error       TEXT,
    failed_at   INTEGER
);
```

**2. Compensating Transactions for Rollback**

Each write operation registers its inverse (compensating transaction). If a multi-step write fails halfway, the compensating transactions for completed steps are replayed in reverse order.

```python
# core/store/unified.py

class MemoryStore:
    async def write(self, memory: Memory) -> str:
        """
        Transactional write across all three stores.
        SQLite is written synchronously (source of truth).
        Qdrant and Kuzu are written via outbox (eventually consistent).
        """
        async with self.sqlite.transaction():
            # 1. Write to SQLite (atomic, immediate)
            await self.meta.insert_memory(memory)
            await self.meta.insert_chunks(memory.chunks)
            await self.meta.insert_forward_annotations(memory.annotations)
            
            # 2. Enqueue derived writes to outbox
            for chunk in memory.chunks:
                await self.meta.enqueue_outbox(
                    operation='UPSERT_VECTOR',
                    payload={'chunk_id': chunk.id, 
                             'vector': chunk.embedding.tolist(),
                             'payload': chunk.metadata}
                )
            for edge in memory.graph_edges:
                await self.meta.enqueue_outbox(
                    operation='UPSERT_EDGE',
                    payload={'from': edge.from_id, 'to': edge.to_id, 
                             'type': edge.type, 'weight': edge.weight}
                )
        # SQLite transaction commits here — durable even if Qdrant is down
        return memory.id

    async def delete(self, memory_id: str) -> None:
        """
        Deletion follows the same pattern.
        SQLite marks as DELETED (soft delete), outbox queues hard deletes.
        """
        async with self.sqlite.transaction():
            await self.meta.soft_delete(memory_id)   # sets deleted_at
            await self.meta.enqueue_outbox('DELETE_VECTOR', {'memory_id': memory_id})
            await self.meta.enqueue_outbox('DELETE_NODE', {'memory_id': memory_id})


# core/store/outbox_worker.py

class OutboxWorker:
    """
    Background asyncio task. Reads PENDING outbox entries,
    applies to Qdrant/Kuzu, marks DONE or increments attempts.
    """
    MAX_ATTEMPTS = 5
    BACKOFF_BASE = 2  # seconds
    
    async def run(self):
        while True:
            rows = await self.meta.get_pending_outbox(limit=50)
            for row in rows:
                await self._process(row)
            await asyncio.sleep(0.1)  # 100ms polling interval
    
    async def _process(self, row: OutboxRow):
        try:
            await self.meta.mark_in_flight(row.id)
            if row.operation == 'UPSERT_VECTOR':
                await self.vector.upsert(**row.payload)
            elif row.operation == 'DELETE_VECTOR':
                await self.vector.delete(**row.payload)
            elif row.operation == 'UPSERT_EDGE':
                await self.graph.upsert_edge(**row.payload)
            elif row.operation == 'DELETE_NODE':
                await self.graph.delete_node(**row.payload)
            await self.meta.mark_done(row.id)
        except Exception as e:
            attempts = row.attempts + 1
            if attempts >= self.MAX_ATTEMPTS:
                await self.meta.mark_dead(row.id, str(e))
            else:
                backoff = self.BACKOFF_BASE ** attempts
                await self.meta.mark_pending_retry(row.id, attempts, backoff)
```

**3. Read Consistency Strategy**

During the outbox propagation window, Qdrant/Kuzu may lag SQLite by milliseconds. Reads use a **read-your-writes** fallback:

```python
async def search(self, query_vector, top_k=20):
    # 1. Query Qdrant (fast, usually consistent)
    vector_results = await self.vector.search(query_vector, top_k)
    
    # 2. Check SQLite for any memories written in last 5s not yet in Qdrant
    #    (handles the propagation lag window)
    recent_ids = await self.meta.get_recently_written(within_seconds=5)
    vector_ids = {r.id for r in vector_results}
    missing = [id for id in recent_ids if id not in vector_ids]
    
    if missing:
        # Fallback: retrieve from SQLite directly and rank by cosine sim
        fallback = await self.meta.get_chunks_by_ids(missing)
        fallback_ranked = cosine_rank(query_vector, fallback)
        vector_results = merge_ranked(vector_results, fallback_ranked, top_k)
    
    return vector_results
```

**4. Rebuild Index Capability**

Since SQLite is the source of truth, add a CLI command to fully rebuild Qdrant and Kuzu from SQLite at any time:

```bash
ppm rebuild-index --store vector   # rebuild Qdrant from SQLite chunks
ppm rebuild-index --store graph    # rebuild Kuzu from SQLite edges
ppm rebuild-index --all            # full rebuild
```

This is the escape hatch. If the derived stores get corrupted, you recover in minutes.

---

## UPDATED COMPONENT SUMMARY

| Component | Before | After |
|---|---|---|
| Token budget | Fractional knapsack + truncation | 0/1 DP knapsack on pre-chunked semantic units |
| Chunking | At injection time (dangerous) | At write time (AST/sentence boundaries) |
| Momentum math | Euclidean extrapolation | Geodesic SLERP on unit hypersphere |
| Vector normalization | Not guaranteed | `UnitVector` type enforced at all module boundaries |
| State window | Fixed N, fixed λ | Hard cap N=6, adaptive λ with context-switch reset |
| Cross-store writes | Fire-and-forget (inconsistent) | Outbox pattern, SQLite as source of truth |
| Store failures | Silently corrupt | Retry with backoff → dead letter → rebuild CLI |
| Read consistency | Potentially stale | Read-your-writes fallback within 5s window |

---

## UPDATED CLAUDE CODE PROMPT ADDENDUM

Append this to the original prompt:

```
Additional constraints from v2 refinement:

1. CHUNKING: All memories must be chunked at write time using SemanticChunker 
   (AST boundaries for code, sentence boundaries for prose). The knapsack 
   operates on chunks, never raw memory content. Content is NEVER truncated.

2. MATH: All embedding vectors must be L2-normalized (UnitVector type) at 
   module boundaries. Use geodesic SLERP extrapolation in predict_future_state(), 
   not linear addition. Add a test asserting ||predict_future_state(...)|| ≈ 1.0.

3. STATE WINDOW: Hard cap the lookback window at N=6. Implement adaptive_decay() 
   in math/entropy.py with context-switch detection. ConversationObserver must 
   call adaptive_decay() on every turn and reset momentum if did_reset=True.

4. STORE WRITES: Implement the outbox pattern in core/store/unified.py. 
   SQLite is the source of truth. All Qdrant and Kuzu writes go through the 
   outbox table. Implement OutboxWorker as a background asyncio task. 
   Add `ppm rebuild-index` CLI command.
```
