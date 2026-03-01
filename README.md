<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/status-alpha-orange?style=for-the-badge" />
</p>

<h1 align="center">presage</h1>

<p align="center">
  <b>Anticipatory memory for LLMs. Retrieves before you ask.</b><br/>
  <sub>A novel memory architecture that predicts what context an LLM will need next — and pre-stages it before the question arrives.</sub>
</p>

---

Every memory system for LLMs works the same way: the user asks something, the system retrieves relevant context, the LLM responds. Retrieve, then respond.

**Presage inverts this.** It models the conversation as a trajectory moving through semantic space — with position, velocity, and acceleration — and uses that trajectory to predict what memories will be needed *next*. By the time the user sends their next message, the relevant context is already staged and ready to inject.

No retrieval latency on the critical path. No cold cache. A system that gets smarter every turn.

---

## How It Works

```
User types a message
        │
        ├── [HOT PATH — blocks until response] ──────────────────────┐
        │   Observer:  embed + intent extraction        (~2ms)        │
        │   Staging:   grab pre-fetched memory          (~1ms)        │  
        │   Reranker:  refine against actual message    (~1ms)        │
        │   Injector:  0/1 knapsack → token budget      (~1ms)        │
        │   LLM:       call with enriched context                     │
        │                                                             │
        └── [BACKGROUND — while LLM generates] ──────────────────────┘
            Predictor:  geodesic extrapolation → predictions
            Prefetcher: async fetch from vector + graph + annotations
            Feedback:   hit/miss → bandit update → training log
            Writer:     distill + chunk + store new memories
```

By the next turn, the prefetch is already done. The user never waits for retrieval.

---

## The Math

Presage treats conversation as a particle moving through the embedding hypersphere.

**Conversation state** — an exponentially-decayed weighted sum of turn embeddings, normalized onto the unit sphere:

$$C_t = \text{normalize}\left(\sum_{i=0}^{N} \lambda^{(N-i)} \cdot e_i\right)$$

**Momentum** — projected onto the tangent plane at $C_t$ (respects spherical geometry):

$$M_{\text{tan}} = M_t - (M_t \cdot C_t)\,C_t$$

**Geodesic extrapolation** — moves along the great circle rather than punching through the sphere's interior:

$$\hat{C}_{t+k} = \cos(\theta)\,C_t + \sin(\theta)\,\hat{M}_{\text{tan}}, \quad \theta = v \cdot k \cdot \delta$$

The predicted state $\hat{C}_{t+k}$ is used as the query vector for prefetching — always on the unit sphere, always a valid cosine similarity query.

**Confidence** — each prediction strategy is tracked by a Bayesian Beta-Bernoulli bandit:

$$P(\text{hit}) = \frac{\alpha_{\text{hits}} + 1}{\alpha_{\text{hits}} + \beta_{\text{misses}} + 2}$$

No training required. Starts calibrated (Beta(1,1) = 0.5), updates every turn.

**Injection** — context allocation solved as a 0/1 knapsack over pre-chunked semantic units:

$$\max \sum v_i x_i \quad \text{s.t.} \quad \sum w_i x_i \leq B, \quad x_i \in \{0, 1\}$$

Content is **never truncated**. The knapsack selects whole chunks only — split at AST node, sentence, or header boundaries at write time.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Surface Layer    token stream · intent classifier           │
│  (core/surface)   symbol extractor · file detector          │
├─────────────────────────────────────────────────────────────┤
│  Nerve Layer      conversation state manager                 │
│  (core/nerve)     trajectory predictor · bandit registry    │
├─────────────────────────────────────────────────────────────┤
│  Staging Layer    P0–P9 async prefetch cache                 │
│  (core/staging)   prefetcher · reranker · knapsack injector │
├─────────────────────────────────────────────────────────────┤
│  Store Layer      SQLite (source of truth)                   │
│  (core/store)     Qdrant (vector) · Kuzu (graph)            │
│                   outbox worker · read-your-writes fallback  │
├─────────────────────────────────────────────────────────────┤
│  Write Layer      memory distiller · conflict resolver       │
│  (core/write)     semantic chunker · forward annotator      │
├─────────────────────────────────────────────────────────────┤
│  Feedback Layer   trigram + semantic hit detection           │
│  (core/feedback)  bandit updater · trajectory dataset       │
├─────────────────────────────────────────────────────────────┤
│  API Layer        FastAPI REST · WebSocket streaming         │
│  (api/)           session factory · session manager         │
└─────────────────────────────────────────────────────────────┘
```

### Storage

| Store | Backend | Role |
|---|---|---|
| MetaStore | SQLite + aiosqlite | Source of truth. All writes here first. |
| VectorStore | Qdrant (local) | Semantic search over chunk embeddings. |
| GraphStore | Kuzu (embedded) | Causal graph: calls, imports, conflicts. |
| Outbox | SQLite table | Eventual consistency to Qdrant + Kuzu. |

**SQLite is always the source of truth.** Qdrant and Kuzu are derived projections — if they get corrupted, run `presage rebuild-index` to reconstruct them from SQLite in minutes.

---

## Quickstart

### Prerequisites

- Python 3.12+
- An Anthropic or OpenAI API key (or Ollama for local models)

### Install

```bash
pip install presage
```

Or with all optional backends:

```bash
pip install "presage[all]"
```

### Initialize

```bash
presage init
```

### Ingest your codebase

```bash
presage ingest ./your_project/
```

Presage walks the directory, chunks every file at natural boundaries (AST nodes for code, headers for markdown, top-level keys for JSON/YAML), embeds the chunks, and writes them to the store with forward annotations.

### Chat

```bash
presage chat
```

```
Session: a3f7c2d1-...
Type your message. Ctrl+C to exit.

You: why does verify_token throw an AttributeError?

Presage [DEBUG | v=0.18 | 3 mem | 241ms]:
The AttributeError in verify_token() is caused by...
```

The header shows: detected intent, conversation velocity, memories injected, and turn latency.

### Start the API

```bash
presage serve
```

API docs: `http://localhost:8000/docs`

---

## REST API

```bash
# Create a session
curl -X POST http://localhost:8000/v1/session

# Submit a turn
curl -X POST http://localhost:8000/v1/turn \
  -H "Content-Type: application/json" \
  -d '{"session_id": "...", "message": "explain verify_token"}'

# Manually ingest a memory
curl -X POST http://localhost:8000/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"content": "...", "source": "auth.py", "source_type": "code"}'

# Search memories
curl "http://localhost:8000/v1/memory/search?query=authentication&top_k=5"

# View staging slot state
curl http://localhost:8000/v1/session/{id}/slots

# Health check
curl http://localhost:8000/v1/health
```

---

## Docker

```bash
cd docker
cp .env.example .env   # add API keys
docker-compose up -d
```

| Service | URL | Purpose |
|---|---|---|
| Presage API | `http://localhost:8000` | REST + WebSocket |
| API Docs | `http://localhost:8000/docs` | Swagger UI |
| Metrics | `http://localhost:8000/metrics` | Prometheus scrape |
| Grafana | `http://localhost:3000` | Metrics dashboard |

---

## Configuration

All settings are environment variables with the `PRESAGE_` prefix (or set in `.env`).

| Variable | Default | Description |
|---|---|---|
| `PRESAGE_LLM_BACKEND` | `anthropic` | `anthropic` · `openai` · `ollama` |
| `PRESAGE_EMBEDDER_BACKEND` | `openai` | `openai` · `nomic` · `bge` |
| `PRESAGE_DECAY_LAMBDA_BASE` | `0.85` | Exponential decay for conversation state |
| `PRESAGE_CONTEXT_SWITCH_THRESHOLD` | `0.40` | Cosine distance that triggers momentum reset |
| `PRESAGE_SLERP_STEP_SIZE` | `0.30` | Arc length per velocity unit in geodesic extrapolation |
| `PRESAGE_AUTO_INJECT_THRESHOLD` | `0.80` | Bandit confidence required for automatic injection |
| `PRESAGE_MAX_INJECT_TOKENS` | `4096` | Token budget for context injection per turn |
| `PRESAGE_SLOT_COUNT` | `10` | Number of prefetch staging slots (P0–P9) |
| `PRESAGE_SLOT_TTL_SECONDS` | `120` | How long a staged memory stays warm |
| `PRESAGE_STATE_WINDOW_MAX` | `6` | Max turn lookback for conversation state |

---

## Prediction Strategies

Presage uses different retrieval strategies depending on detected intent:

| Intent | Signals | Strategy |
|---|---|---|
| `DEBUG` | "error", "fix", "crash", "exception" | Graph walk → semantic |
| `IMPLEMENT` | "write", "create", "build", "add" | Semantic → symbol lookup |
| `NAVIGATE` | "where is", "find", "which file" | Symbol lookup → semantic |
| `COMPARE` | "vs", "difference", "better than" | Hybrid (vector + graph) |
| `EXPLORE` | "what is", "explain", "how does" | Semantic → annotation |
| `REFLECT` | "earlier", "we decided", "before" | Annotation → semantic |

---

## Staging Slots

The 10 staging slots are tiered by confidence:

```
P0 ─── AUTO  (conf ≥ 0.80) → injected automatically every turn
P1 ─── AUTO  (conf ≥ 0.80) → injected automatically every turn
P2 ─── HOT   (conf ≥ 0.50) → injected when soft trigger matches
P3 ─── HOT   (conf ≥ 0.50) → injected when soft trigger matches
P4 ─── HOT   (conf ≥ 0.50) → injected when soft trigger matches
P5 ─── WARM  (conf ≥ 0.30) → available on explicit request
...
P9 ─── WARM  (conf ≥ 0.30) → available on explicit request
```

A soft trigger fires when the user's message mentions a symbol or file that matches a staged memory's annotation tags — e.g., typing "verify_token" fires any HOT memory tagged `symbol:verify_token`.

---

## Benchmarks

```bash
python tests/bench/bench_momentum.py   # math layer
python tests/bench/bench_staging.py    # staging layer
python tests/bench/bench_store.py      # store layer
```

Target latencies (P99 on a modern laptop):

| Operation | Target | Layer |
|---|---|---|
| Intent classification | < 0.5ms | Surface |
| Momentum update | < 2ms | Nerve |
| Geodesic extrapolation | < 2ms | Nerve |
| Knapsack injection | < 1ms | Staging |
| Reranker | < 1ms | Staging |
| Annotation search | < 5ms | Store |
| Total hot-path overhead | **< 10ms** | All |

The prefetch (retrieval) runs in the background while the LLM generates its response — it does not contribute to user-perceived latency.

---

## Observability

Presage exposes Prometheus metrics at `/metrics` and OpenTelemetry traces via OTLP.

Key metrics:

```
presage_session_turn_latency_seconds    # end-to-end hot path latency
presage_feedback_hit_rate               # prediction hit rate per turn
presage_nerve_momentum_velocity         # conversation velocity histogram
presage_staging_slot_hits_total         # successful memory injections
presage_store_outbox_pending            # propagation lag gauge
presage_trajectory_samples_total        # training data accumulated
```

Enable tracing:
```bash
PRESAGE_OTEL_ENDPOINT=http://jaeger:4317 presage serve
```

---

## Training Data Export

Every session accumulates trajectory samples — `(conversation_state, predictions, outcomes)` triples — that can be used to fine-tune the trajectory predictor from heuristic rules into a learned model.

```bash
presage export trajectory_data.jsonl
```

The JSONL format is compatible with standard fine-tuning pipelines. Export requires sessions with ≥ 100 turns for quality filtering.

---

## Project Structure

```
presage/
├── math_core/           # Core mathematics
│   ├── momentum.py      # Conversation state, SLERP extrapolation
│   ├── entropy.py       # Context switch detection, adaptive decay
│   ├── knapsack.py      # 0/1 DP knapsack for token budget
│   ├── diffusion.py     # Personalized PageRank over memory graph
│   └── bandit.py        # Beta-Bernoulli bandits + registry
├── core/
│   ├── surface/         # Observer, intent classifier, signal extractor
│   ├── nerve/           # Trajectory predictor, state manager
│   ├── staging/         # Prefetch cache, prefetcher, injector, reranker
│   ├── store/           # MetaStore, VectorStore, OutboxWorker
│   ├── write/           # Chunker, distiller, conflict resolver, annotator
│   ├── feedback/        # Hit/miss detector, feedback loop, dataset
│   └── session/         # SessionManager, SessionFactory
├── adapters/
│   ├── embedder/        # OpenAI, nomic, bge (local)
│   └── llm/             # Anthropic, OpenAI, Ollama
├── api/                 # FastAPI REST + WebSocket
├── observability/       # Prometheus metrics, OpenTelemetry tracing
├── cli/                 # presage CLI
├── docker/              # Dockerfile, docker-compose, .env.example
└── tests/
    ├── unit/            # Per-module unit tests (100% coverage target)
    ├── integration/     # Cross-layer integration tests
    └── bench/           # Latency benchmarks
```

---

## What Makes Presage Different

Every existing LLM memory system — Mem0, MemGPT, Zep, LangChain memory, AriGraph — is **pull-based**. The LLM asks, the store answers.

Presage is **push-based**. The store predicts, prefetches, and pushes. By the time the LLM asks, the answer is already there.

| System | Architecture | Retrieval | Self-improving |
|---|---|---|---|
| RAG | Vector search | Reactive | No |
| MemGPT | Episodic compression | Reactive | No |
| Zep | Bi-temporal KG | Reactive | No |
| A-MEM | Zettelkasten | Reactive | No |
| **Presage** | **Kinematic trajectory** | **Proactive** | **Yes (bandits)** |

---

## Roadmap

- [ ] Cross-encoder reranker for P0-P1 slots (Phase 7 upgrade)
- [ ] Token streaming in WebSocket (`type: token` events)
- [ ] Fine-tuned trajectory predictor from accumulated dataset
- [ ] Multi-user shared memory with access control
- [ ] VSCode extension for native IDE integration
- [ ] MCP server adapter (plug into Claude, Cursor, Zed)

---

## License

MIT © 2025

---

<p align="center">
  <sub>Built with kinematic trajectory math, Bayesian bandits, and the conviction that memory should anticipate — not react.</sub>
</p>