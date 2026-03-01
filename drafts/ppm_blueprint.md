# PPM: Predictive Push Memory
## Full Architecture Blueprint — Claude Code Ready
> A self-improving, mathematically-grounded, anticipatory memory brain for LLMs

---

## 0. GUIDING PRINCIPLES

| Principle | Implementation Strategy |
|---|---|
| **Fast** | Async prefetch hides latency entirely. All hot-path ops < 5ms. |
| **Scalable** | Every layer is independently horizontally scalable. Stateless services. |
| **Adaptable** | Plugin-based adapters for storage, embedders, and LLM backends. |
| **Maintainable** | Strict layer separation. Each module has one responsibility. Full observability. |

---

## 1. REPOSITORY STRUCTURE

```
ppm/
├── core/
│   ├── surface/          # Token stream observation & intent extraction
│   ├── nerve/            # Trajectory math & prediction engine
│   ├── staging/          # Hot cache & slot manager
│   ├── store/            # Storage backends (vector, graph, kv)
│   ├── write/            # Memory formation & distillation
│   └── feedback/         # Hit/miss logging & predictor fine-tuning
├── math/
│   ├── momentum.py       # Momentum vector algebra
│   ├── diffusion.py      # Relevance diffusion over graph
│   ├── entropy.py        # Conversation entropy & velocity
│   └── bandit.py         # Multi-armed bandit slot allocation
├── adapters/
│   ├── llm/              # OpenAI, Anthropic, Ollama, etc.
│   ├── embedder/         # OpenAI, local (nomic, bge), etc.
│   └── storage/          # Qdrant, Chroma, Kuzu, SQLite, Redis
├── api/
│   ├── server.py         # FastAPI server
│   └── ws.py             # WebSocket for real-time token streaming
├── cli/
│   └── ppm.py            # CLI entry point
├── config/
│   └── settings.py       # Pydantic settings — fully env-driven
└── tests/
    ├── unit/
    ├── integration/
    └── bench/            # Latency & throughput benchmarks
```

---

## 2. THE CORE MATHEMATICS

This is what makes PPM novel. Each concept maps directly to code.

### 2.1 Conversation State Vector

At any turn `t`, the conversation is represented as a **state vector** in a high-dimensional semantic space:

```
C_t ∈ ℝ^d

C_t = Σ_{i=0}^{N} λ^(t-i) · embed(turn_i)
```

Where:
- `d` = embedding dimension (e.g., 768 or 1536)
- `λ ∈ (0,1)` = **exponential decay factor** (recent turns weighted more)
- `embed(turn_i)` = dense embedding of turn i
- `N` = lookback window (last N turns)

This gives a single vector that represents "where we are" in semantic space, biased toward recency.

```python
# math/momentum.py
import numpy as np

def conversation_state(turn_embeddings: list[np.ndarray], decay: float = 0.85) -> np.ndarray:
    """
    Exponentially decayed weighted sum of turn embeddings.
    O(N·d) time, O(d) space.
    """
    if not turn_embeddings:
        return np.zeros(turn_embeddings[0].shape)
    weights = np.array([decay ** i for i in range(len(turn_embeddings))][::-1])
    weights /= weights.sum()  # normalize
    stacked = np.stack(turn_embeddings)  # (N, d)
    return weights @ stacked  # (d,)
```

---

### 2.2 Momentum Vector (The Core Innovation)

The **momentum vector** is the first derivative of the conversation state — the *direction and speed* of semantic drift:

```
M_t = C_t - C_{t-1}           (instantaneous momentum)

M_t^smooth = β·M_{t-1} + (1-β)·M_t    (smoothed, like Adam optimizer's m̂)
```

The **magnitude** `||M_t||` is **topic velocity** — how fast are we moving through semantic space:
- High velocity → aggressive prefetch (conversation is shifting)
- Low velocity → focused, narrow prefetch (user is going deep)

The **direction** `M_t / ||M_t||` is **where we are heading**.

```python
def momentum(C_prev: np.ndarray, C_curr: np.ndarray,
             M_prev: np.ndarray, beta: float = 0.9) -> tuple[np.ndarray, float]:
    """Returns smoothed momentum vector and scalar velocity."""
    raw_momentum = C_curr - C_prev
    smoothed = beta * M_prev + (1 - beta) * raw_momentum
    velocity = float(np.linalg.norm(smoothed))
    return smoothed, velocity
```

---

### 2.3 Predictive Target Distribution

Given the current state `C_t` and momentum `M_t`, we project **where the conversation will be** in `k` steps:

```
Ĉ_{t+k} = C_t + k·M_t^smooth + ½·k²·A_t
```

Where `A_t = M_t - M_{t-1}` is the **semantic acceleration** (second derivative — are we speeding up or changing direction?).

This is a **kinematic extrapolation** in embedding space — borrowed from classical mechanics.

The **prefetch query** is then:

```
q_prefetch = Ĉ_{t+1}          # project 1 step ahead (conservative)
q_prefetch = Ĉ_{t+2}          # project 2 steps (aggressive)
```

Aggressiveness is gated by velocity: `k = ceil(velocity / velocity_threshold)`.

```python
def predict_future_state(C_t, M_smooth, A_t=None, k: int = 1) -> np.ndarray:
    """Kinematic extrapolation in embedding space."""
    pred = C_t + k * M_smooth
    if A_t is not None:
        pred += 0.5 * (k ** 2) * A_t
    return pred  # use as query vector for memory retrieval
```

---

### 2.4 Relevance Diffusion over the Memory Graph

Once a seed memory is retrieved by vector search, we expand it through the causal/semantic graph using **personalized PageRank** (heat diffusion):

```
r^(0) = e_seed          (one-hot on seed node)
r^(t+1) = α·W·r^(t) + (1-α)·e_seed
```

Where:
- `W` = row-normalized adjacency matrix of the memory graph
- `α = 0.85` = damping factor (standard PageRank)
- Converges in ~10 iterations

This surfaces **related memories the user didn't explicitly ask for** — e.g., you query `auth.js` and get back `session.js`, `User.js`, and the auth test file automatically.

```python
# math/diffusion.py
def personalized_pagerank(adj: dict[str, list[str]], seed_id: str,
                           alpha: float = 0.85, iters: int = 10) -> dict[str, float]:
    """Sparse PPR. Returns {node_id: relevance_score}."""
    nodes = list(adj.keys())
    scores = {n: 0.0 for n in nodes}
    scores[seed_id] = 1.0
    for _ in range(iters):
        new_scores = {n: 0.0 for n in nodes}
        for node, neighbors in adj.items():
            if not neighbors:
                continue
            share = alpha * scores[node] / len(neighbors)
            for nb in neighbors:
                new_scores[nb] += share
        # teleport
        new_scores[seed_id] += (1 - alpha)
        scores = new_scores
    return scores
```

---

### 2.5 Confidence Scoring (Bayesian)

Each prediction slot has a **Bayesian confidence score** built from prior hit rate:

```
P(hit | context) = (α_hits + 1) / (α_hits + β_misses + 2)    [Laplace smoothed Beta distribution]
```

Updated after every turn:
- Hit: `α_hits += 1`
- Miss: `β_misses += 1`

The slot injection threshold: `P(hit) > θ_inject` (default 0.6).
The staging threshold: `P(hit) > θ_stage` (default 0.3).

This means the predictor is **always calibrated** — it knows its own reliability.

```python
# math/bandit.py
class BetaBandit:
    """Per-prediction-type Bayesian hit tracker."""
    def __init__(self):
        self.alpha = 1.0  # prior hits
        self.beta = 1.0   # prior misses

    def confidence(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, hit: bool):
        if hit:
            self.alpha += 1
        else:
            self.beta += 1

    def sample(self) -> float:
        """Thompson sampling for exploration."""
        import random
        # approximate beta distribution sample
        return random.betavariate(self.alpha, self.beta)
```

---

### 2.6 Context Budget Allocation (Knapsack)

Given remaining context tokens `B` and `N` staged memories each with weight `w_i` (tokens) and value `v_i` (confidence × relevance), we solve a **fractional knapsack**:

```
maximize   Σ v_i · x_i
subject to Σ w_i · x_i ≤ B
           0 ≤ x_i ≤ 1
```

Fractional knapsack has an O(N log N) greedy solution: sort by `v_i/w_i` descending, fill greedily. This guarantees optimal memory injection under the token budget.

```python
def fractional_knapsack(items: list[dict], budget: int) -> list[dict]:
    """
    items: [{'id': ..., 'tokens': int, 'score': float, 'content': str}]
    Returns selected items (possibly truncated last item).
    """
    sorted_items = sorted(items, key=lambda x: x['score'] / max(x['tokens'], 1), reverse=True)
    selected, remaining = [], budget
    for item in sorted_items:
        if remaining <= 0:
            break
        if item['tokens'] <= remaining:
            selected.append(item)
            remaining -= item['tokens']
        else:
            # fractional: truncate content to fit
            ratio = remaining / item['tokens']
            truncated = dict(item)
            truncated['content'] = item['content'][:int(len(item['content']) * ratio)]
            truncated['tokens'] = remaining
            selected.append(truncated)
            remaining = 0
    return selected
```

---

## 3. MODULE SPECIFICATIONS

### 3.1 Surface Layer — `core/surface/`

**Responsibility:** Observe the conversation stream and extract signals.

```python
# core/surface/observer.py
class ConversationObserver:
    """
    Streams tokens in real-time. Extracts:
    - Turn embeddings (batched, async)
    - Intent signals (heuristic + classifier)
    - Entropy estimate (surprise = log(1/p_token))
    """
    async def on_token(self, token: str) -> None: ...
    async def on_turn_complete(self, turn: Turn) -> TurnSignals: ...
```

**Intent Signal Taxonomy (heuristic rules → classifier later):**

| Signal | Trigger Phrases | Prefetch Bias |
|---|---|---|
| `EXPLORE` | "what is", "explain", "tell me about" | Broad semantic search |
| `DEBUG` | "why does", "fix", "error", "not working" | Graph walk: caller + callee + test |
| `NAVIGATE` | "where is", "show me", "find" | Symbol index lookup |
| `COMPARE` | "difference between", "vs", "better" | Dual retrieval |
| `IMPLEMENT` | "how do I", "write a", "create" | Examples + docs + similar functions |
| `REFLECT` | "what did we", "earlier you", "before" | Episodic memory retrieval |

---

### 3.2 Nerve Layer — `core/nerve/`

**Responsibility:** Run the trajectory math and output predictions.

```python
# core/nerve/predictor.py
class TrajectoryPredictor:
    def __init__(self, embedder: Embedder, history_size: int = 10):
        self.embedder = embedder
        self.state_history: deque[np.ndarray] = deque(maxlen=history_size)
        self.momentum: np.ndarray = None
        self.bandits: dict[str, BetaBandit] = defaultdict(BetaBandit)

    async def update(self, turn: Turn) -> list[Prediction]:
        embedding = await self.embedder.embed(turn.text)
        C_t = conversation_state([*self.state_history, embedding])
        M_t, velocity = momentum(self.state_history[-1], C_t, self.momentum)

        future = predict_future_state(C_t, M_t, k=max(1, int(velocity * 3)))
        predictions = self._generate_predictions(future, turn.intent_signals, velocity)
        
        self.state_history.append(embedding)
        self.momentum = M_t
        return predictions

    def _generate_predictions(self, future_state, signals, velocity) -> list[Prediction]:
        """
        Returns ranked predictions. Each prediction has:
        - query_vector: np.ndarray (for vector search)
        - query_text: str (for keyword search)
        - graph_seeds: list[str] (for graph diffusion)
        - confidence: float
        - strategy: str (SEMANTIC | GRAPH | SYMBOL | HYBRID)
        """
        ...
```

---

### 3.3 Staging Layer — `core/staging/`

**Responsibility:** Async prefetch, slot management, injection decisions.

```python
# core/staging/cache.py
class StagingCache:
    """
    10 prioritized slots. Async background prefetcher.
    Thread-safe via asyncio locks.
    """
    SLOT_COUNT = 10
    
    # Slot tiers
    AUTO_INJECT_THRESHOLD = 0.80    # slots 0-1: auto inject
    HOT_THRESHOLD = 0.50            # slots 2-4: inject on soft trigger  
    WARM_THRESHOLD = 0.30           # slots 5-9: pre-embedded, instant access
    
    async def prefetch(self, predictions: list[Prediction]) -> None:
        """Runs in background. Non-blocking to main conversation loop."""
        async with asyncio.TaskGroup() as tg:
            for pred in predictions[:self.SLOT_COUNT]:
                tg.create_task(self._fill_slot(pred))
    
    async def get_injectable(self, budget_tokens: int, 
                              trigger: str | None = None) -> list[Memory]:
        """Returns memories to inject. Respects token budget via knapsack."""
        candidates = self._get_candidates(trigger)
        return fractional_knapsack(candidates, budget_tokens)
    
    def evict_expired(self) -> None:
        """TTL-based eviction. Called after each turn."""
        now = time.time()
        self.slots = [s for s in self.slots if now - s.created_at < s.ttl]
```

---

### 3.4 Store Layer — `core/store/`

**Responsibility:** Persistent memory. Three sub-stores, one unified interface.

```python
# core/store/unified.py
class MemoryStore:
    """
    Unified interface over three backends:
    - VectorStore (Qdrant): semantic search
    - GraphStore (Kuzu): relationship traversal
    - MetaStore (SQLite): structured queries & forward annotations
    """
    async def search(self, query_vector: np.ndarray, top_k: int = 20,
                     filter: dict | None = None) -> list[Memory]: ...
    
    async def graph_walk(self, seed_ids: list[str], 
                          depth: int = 2) -> list[Memory]: ...
    
    async def symbol_lookup(self, symbol: str) -> list[Memory]: ...
    
    async def forward_annotation_search(self, context_tags: list[str]) -> list[Memory]: ...
    
    async def write(self, memory: Memory) -> str: ...  # returns memory_id
    
    async def update(self, memory_id: str, new_content: str) -> None: ...
```

**Forward Annotations Schema (SQLite):**

```sql
CREATE TABLE forward_annotations (
    memory_id     TEXT NOT NULL,
    context_tag   TEXT NOT NULL,   -- e.g., "topic:authentication", "intent:DEBUG"
    weight        REAL DEFAULT 1.0,
    created_at    INTEGER,
    hit_count     INTEGER DEFAULT 0,
    PRIMARY KEY (memory_id, context_tag)
);

CREATE INDEX idx_fa_tag ON forward_annotations(context_tag, weight DESC);
```

**Memory Schema:**

```sql
CREATE TABLE memories (
    id            TEXT PRIMARY KEY,
    content       TEXT NOT NULL,
    content_hash  TEXT NOT NULL,  -- for dedup
    source        TEXT,           -- file path, url, conversation_id
    source_type   TEXT,           -- 'code', 'doc', 'conversation', 'distilled'
    token_count   INTEGER,
    created_at    INTEGER,
    last_accessed INTEGER,
    access_count  INTEGER DEFAULT 0,
    version       INTEGER DEFAULT 1,
    parent_id     TEXT            -- for conflict versioning
);
```

**Graph Edge Schema (Kuzu):**

```
Node: Memory(id STRING, type STRING)
Edge: CALLS(from Memory, to Memory)
Edge: IMPORTS(from Memory, to Memory)
Edge: RELATED_TO(from Memory, to Memory, weight DOUBLE)
Edge: CONFLICTS_WITH(from Memory, to Memory, reason STRING)
Edge: SUMMARIZES(from Memory, to Memory)
```

---

### 3.5 Write Layer — `core/write/`

**Responsibility:** Distill and store new memories after each turn.

```python
# core/write/distiller.py
class MemoryDistiller:
    """
    After each LLM response, runs a lightweight pass to decide what to store.
    Uses a small prompt, NOT the full LLM — keep this fast.
    """
    DISTILL_PROMPT = """
    Given this conversation turn, extract ONLY information worth remembering long-term.
    Return JSON: {"worth_storing": bool, "memories": [{"content": str, "tags": list[str], 
    "forward_contexts": list[str], "source_type": str}]}
    If nothing is worth storing, return {"worth_storing": false, "memories": []}.
    Turn: {turn}
    """
    
    async def distill(self, turn: Turn) -> list[MemoryCandidate]: ...
    async def resolve_conflicts(self, new: Memory, existing: list[Memory]) -> ConflictResolution: ...
    async def write_forward_annotations(self, memory: Memory, contexts: list[str]) -> None: ...
```

**Conflict Resolution Strategy:**

```
New memory vs existing:
  ├── Same hash → deduplicate, update access_count
  ├── Contradicts (semantic similarity > 0.85, content differs) 
  │   → Create conflict node, keep both, tag with CONFLICTS_WITH edge
  ├── Extends (similarity 0.6-0.85)
  │   → Create new version, link with parent_id, deprecate old
  └── Novel (similarity < 0.6)
      → Store as new independent memory
```

---

### 3.6 Feedback Layer — `core/feedback/`

**Responsibility:** Measure prediction quality, improve the predictor.

```python
# core/feedback/tracker.py
class HitMissTracker:
    """
    After each turn, checks: was any staged memory cited/used by the LLM?
    Detection strategy: 
      1. Check if any prefetched content appears in LLM response (string overlap)
      2. Check if LLM asked for something that was already staged (prevented retrieval)
      3. Proxy: did the LLM produce output strongly semantically similar to staged content?
    """
    async def evaluate_turn(self, response: str, staged: list[Memory],
                             injected: list[Memory]) -> HitMissResult: ...
    
    async def update_bandits(self, result: HitMissResult) -> None: ...
    
    async def log_trajectory_sample(self, state: ConversationState, 
                                     used_memories: list[str]) -> None:
        """Accumulates training data for future predictor fine-tuning."""
        ...
```

**Training Data Format (for future fine-tuning):**

```json
{
  "conversation_state_embedding": [...],
  "momentum_vector": [...],
  "velocity": 0.34,
  "intent_signals": ["DEBUG", "NAVIGATE"],
  "memories_that_were_actually_used": ["mem_abc123", "mem_xyz789"],
  "memories_that_were_prefetched_but_unused": ["mem_def456"],
  "timestamp": 1700000000
}
```

---

## 4. DATA FLOW (End-to-End)

```
USER MESSAGE ARRIVES
│
├──[SYNC, hot path]──────────────────────────────────────────────────┐
│  1. Surface: extract intent signals (< 1ms, heuristic)             │
│  2. Nerve: update momentum vector (< 2ms, numpy ops)               │
│  3. Staging: pull auto-inject slots (confidence > 0.8) (< 1ms)    │
│  4. Knapsack: fit staged memories into context budget (< 1ms)      │
│  5. LLM: call with enriched context                                 │
│  6. Stream response to user                                         │
└──[ASYNC, background]───────────────────────────────────────────────┘
   A. Nerve: generate predictions from momentum+intent
   B. Staging: prefetch top predictions from store (parallel)
   C. Graph: run PPR diffusion from seed results
   D. Stage: fill slots with results + confidence scores

AFTER LLM RESPONSE
├── Feedback: evaluate hit/miss
├── Feedback: update bandits & log trajectory sample  
├── Write: distill new memories from turn
├── Write: resolve conflicts
├── Write: update forward annotations
└── Staging: evict expired slots, re-rank remaining
```

---

## 5. API DESIGN

### REST (FastAPI)

```
POST   /v1/session                    Create new session
DELETE /v1/session/{id}               End session

POST   /v1/turn                       Submit turn, get enriched context
GET    /v1/turn/{id}/staged           See what's staged for next turn
GET    /v1/turn/{id}/hits             Get hit/miss report

POST   /v1/memory                     Manually write a memory
GET    /v1/memory/search?q=...        Search memories
DELETE /v1/memory/{id}                Delete a memory

GET    /v1/stats                      Predictor accuracy, latency P50/P95/P99
GET    /v1/health                     Health check
```

### WebSocket (real-time token stream)

```
WS /v1/stream

Client → Server: {"type": "token", "content": "t", "session_id": "..."}
Client → Server: {"type": "turn_end", "session_id": "..."}
Server → Client: {"type": "staged", "slot": 0, "confidence": 0.91, "preview": "..."}
Server → Client: {"type": "inject", "memories": [...], "tokens_used": 847}
```

---

## 6. PERFORMANCE TARGETS

| Operation | Target Latency | Strategy |
|---|---|---|
| Intent extraction | < 1ms | Pure heuristic regex/keyword rules |
| Momentum update | < 2ms | Numpy vector ops (no GPU needed) |
| Slot injection decision | < 1ms | In-memory sorted list |
| Async prefetch (vector) | < 50ms | Qdrant ANN search, pre-warmed |
| Async prefetch (graph) | < 30ms | Kuzu native graph traversal |
| Memory write | < 20ms | Async, non-blocking to main loop |
| Full turn overhead | **< 5ms** | Everything async except momentum |

---

## 7. TECH STACK

```yaml
language: Python 3.12
async: asyncio + uvloop (2x faster than default loop)

api:
  - FastAPI (REST)
  - websockets (streaming)
  - uvicorn

storage:
  vector: Qdrant (local mode, no docker needed for dev)
  graph:  Kuzu (embedded, zero-config)
  kv:     SQLite (via aiosqlite — async)
  cache:  Redis (optional, for multi-process staging sync)

math:
  - numpy (vectors, momentum)
  - scipy (sparse PPR for large graphs)
  - scikit-learn (cosine similarity, normalization)

embedders (adapter pattern):
  - openai (text-embedding-3-small — fast, cheap)
  - nomic-embed-text (local, via ollama)
  - bge-m3 (local, best quality)

llm adapters:
  - anthropic
  - openai
  - ollama (local models)

observability:
  - structlog (structured logging)
  - prometheus-client (metrics)
  - opentelemetry (traces)

testing:
  - pytest + pytest-asyncio
  - hypothesis (property-based tests for math modules)
```

---

## 8. BUILD ORDER FOR CLAUDE CODE

Build in this exact sequence. Each phase is independently testable.

### Phase 1 — Foundation (Week 1)
```
[x] config/settings.py          — Pydantic settings, env-driven
[x] adapters/embedder/base.py   — Embedder protocol
[x] adapters/embedder/openai.py — OpenAI embedder
[x] math/momentum.py            — conversation_state(), momentum(), predict_future_state()
[x] math/entropy.py             — velocity calculations
[x] core/store/meta.py          — SQLite schema + CRUD
[x] core/store/vector.py        — Qdrant adapter
[x] core/store/graph.py         — Kuzu adapter
[x] core/store/unified.py       — Unified MemoryStore
```

### Phase 2 — Write Pipeline (Week 1-2)
```
[x] core/write/distiller.py     — Memory distillation
[x] core/write/conflict.py      — Conflict resolution
[x] core/write/annotator.py     — Forward annotation writer
```

### Phase 3 — Nerve & Prediction (Week 2)
```
[x] math/bandit.py              — BetaBandit confidence tracking
[x] core/nerve/predictor.py     — TrajectoryPredictor
[x] core/surface/observer.py    — ConversationObserver + intent signals
```

### Phase 4 — Staging & Injection (Week 2-3)
```
[x] math/bandit.py (knapsack)   — fractional_knapsack()
[x] math/diffusion.py           — personalized_pagerank()
[x] core/staging/cache.py       — StagingCache with async prefetch
[x] core/staging/injector.py    — Context injection manager
```

### Phase 5 — Feedback Loop (Week 3)
```
[x] core/feedback/tracker.py    — Hit/miss evaluator
[x] core/feedback/dataset.py    — Trajectory dataset builder
```

### Phase 6 — API & Integration (Week 3-4)
```
[x] api/server.py               — FastAPI REST endpoints
[x] api/ws.py                   — WebSocket token streaming
[x] adapters/llm/base.py        — LLM adapter protocol
[x] adapters/llm/anthropic.py   — Claude integration
[x] cli/ppm.py                  — CLI: init, ingest, chat
```

### Phase 7 — Observability & Polish (Week 4)
```
[x] Prometheus metrics on all layers
[x] OpenTelemetry trace spans across async calls
[x] Benchmark suite: latency P50/P95/P99 per operation
[x] Docker + docker-compose for full local stack
[x] README with architecture diagram
```

---

## 9. KEY DESIGN DECISIONS & RATIONALE

**Why Kuzu over Neo4j?**
Kuzu is an embedded graph DB (like SQLite for graphs). Zero config, no server, stores in a local file. For Phase 1, this is critical for developer experience. Can swap to Neo4j later via the adapter.

**Why exponential decay for conversation state instead of a fixed window?**
A fixed window treats turn N-5 and turn N-1 equally. Exponential decay respects that recent context is more predictive. The decay factor λ is tunable per use case (code: 0.9, general chat: 0.7).

**Why fractional knapsack not 0/1 knapsack?**
0/1 knapsack is NP-hard. Fractional is O(N log N) greedy and gives the optimal solution for divisible items (text chunks are divisible). The last item can be truncated to fill remaining budget exactly.

**Why Bayesian bandits for confidence instead of a neural confidence model?**
Zero training data required at startup. Starts with sensible priors, updates on every turn. Interpretable: you can inspect alpha/beta directly. A neural confidence model can replace it later once enough data is accumulated.

**Why PPR (personalized PageRank) for graph expansion instead of BFS?**
BFS retrieves all nodes within depth k equally. PPR ranks nodes by their weighted relevance to the seed — distant but strongly connected nodes score higher than nearby but weakly connected ones. Critical for large codebases with complex dependency trees.

---

## 10. PROMPT TO PASTE INTO CLAUDE CODE

```
I am building PPM (Predictive Push Memory) — a novel anticipatory memory system for LLMs.

The full architecture blueprint is in the attached document. Please start with Phase 1.

Key constraints:
- Python 3.12, async-first (asyncio), type-annotated throughout
- Every module must have a corresponding test file
- Math modules (momentum.py, diffusion.py, etc.) must have docstrings with the exact equations they implement
- Use the adapter pattern for all external services (embedder, LLM, storage)
- All config via Pydantic Settings (environment variables)
- No hardcoded API keys anywhere

Start with:
1. config/settings.py
2. math/momentum.py (implement conversation_state, momentum, predict_future_state with full docstrings)
3. tests/unit/test_momentum.py (property-based tests with hypothesis)
4. adapters/embedder/base.py + adapters/embedder/openai.py

After each file, confirm the module is complete and ask before proceeding to the next.
```
