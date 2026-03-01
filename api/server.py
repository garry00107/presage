"""
PPM FastAPI Server — REST endpoints.

All endpoints are async and non-blocking.
The session manager handles all business logic;
the API layer only does request validation and response formatting.

Endpoints:
  POST   /v1/session              Create session
  DELETE /v1/session/{id}         Close session
  POST   /v1/turn                 Submit turn, get response
  POST   /v1/ingest               Manually ingest a memory
  GET    /v1/memory/search        Search memories
  GET    /v1/session/{id}/slots   Get current staging slot state
  GET    /v1/stats                System statistics
  GET    /v1/health               Health check
"""

import time
import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.deps import FactoryDep, get_session, set_factory
from api.models import (
    CreateSessionRequest, CreateSessionResponse,
    TurnRequest, TurnResponse, StagedSlot,
    IngestRequest, IngestResponse,
    SearchRequest, SearchResponse, SearchResult,
    StatsResponse, HealthResponse,
)
from observability import metrics
from observability.tracing import init_tracing
from observability.middleware import ObservabilityMiddleware, add_metrics_endpoint

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    init_tracing()
    log.info("ppm_server.starting")
    # factory is set externally via set_factory() before uvicorn starts
    yield
    log.info("ppm_server.shutting_down")
    factory = app.state.factory if hasattr(app.state, "factory") else None
    if factory:
        await factory.close_all()


app = FastAPI(
    title="PPM — Predictive Push Memory",
    description="Anticipatory memory brain for LLMs",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ObservabilityMiddleware)
add_metrics_endpoint(app)


# ── Session endpoints ──────────────────────────────────────────────────────────

@app.post("/v1/session", response_model=CreateSessionResponse)
async def create_session(
    req: CreateSessionRequest,
    factory: FactoryDep,
):
    """Create a new PPM session."""
    session = await factory.create_session(
        session_id=req.session_id,
        restore_bandits=req.restore_bandits,
    )
    metrics.session_active.inc()
    return CreateSessionResponse(
        session_id=session.session_id,
        created_at=time.time(),
    )


@app.delete("/v1/session/{session_id}")
async def close_session(session_id: str, factory: FactoryDep):
    """Close a session and persist its state."""
    await factory.close_session(session_id)
    metrics.session_active.dec()
    return {"status": "closed", "session_id": session_id}


# ── Turn endpoint ──────────────────────────────────────────────────────────────

@app.post("/v1/turn", response_model=TurnResponse)
async def submit_turn(req: TurnRequest, factory: FactoryDep):
    """
    Submit a user message turn. Returns LLM response enriched with memory.

    This is the main endpoint. The session must exist (create it first).
    The response is synchronous — LLM call completes before returning.
    For streaming, use the WebSocket endpoint /v1/ws/{session_id}.
    """
    session = await factory.get_session(req.session_id)
    if session is None:
        # Auto-create session if it doesn't exist
        session = await factory.create_session(session_id=req.session_id)

    result = await session.turn(
        user_message=req.message,
        source=req.source,
    )

    metrics.record_turn(
        intent=result.intent.value,
        latency_s=result.latency_ms / 1000.0,
        tokens=result.tokens_injected,
        memories=result.memories_injected,
    )

    slots = [
        StagedSlot(**s)
        for s in result.staged_slot_summary
    ]

    context_preview = ""
    if result.injection_plan.chunks:
        preview_text = result.injection_plan.context_text
        context_preview = preview_text[:200] + "..." if len(preview_text) > 200 else preview_text

    return TurnResponse(
        session_id=result.session_id,
        turn_index=result.turn_index,
        response=result.llm_response,
        intent=result.intent.value,
        velocity=round(result.velocity, 4),
        tokens_injected=result.tokens_injected,
        memories_injected=result.memories_injected,
        latency_ms=round(result.latency_ms, 1),
        context_preview=context_preview,
        staged_slots=slots,
    )


# ── Memory endpoints ───────────────────────────────────────────────────────────

@app.post("/v1/ingest", response_model=IngestResponse)
async def ingest_memory(req: IngestRequest, factory: FactoryDep):
    """
    Manually ingest a memory into the store.
    Useful for bulk-loading codebases or documents before starting a session.
    """
    # Use factory's shared write pipeline components directly
    from core.write.chunker import SemanticChunker
    from core.write.conflict import ConflictResolver
    from core.write.annotator import ForwardAnnotator

    chunker   = factory._chunker
    annotator = factory._annotator

    import uuid, hashlib
    mid = str(uuid.uuid4())
    chunks = chunker.chunk(req.content, mid, req.source_type)

    # Embed chunks
    chunk_texts = [c.content for c in chunks]
    embeddings  = await factory._embedder.embed_batch(chunk_texts)

    chunks_dicts = []
    for rc, emb in zip(chunks, embeddings):
        d = rc.to_dict()
        d["embedding"] = emb.tolist()
        chunks_dicts.append(d)

    annotations = annotator.annotate(
        memory_id=mid,
        content=req.content,
        source=req.source,
        source_type=req.source_type,
        extra_tags=req.forward_contexts,
    )

    memory_dict = {
        "id": mid,
        "content": req.content,
        "source": req.source,
        "source_type": req.source_type,
        "token_count": sum(c["tokens"] for c in chunks_dicts),
        "chunks": chunks_dicts,
        "forward_contexts": [a.context_tag for a in annotations],
        "graph_edges": [],
    }

    written_id = await factory._meta.insert_memory(memory_dict)

    return IngestResponse(
        memory_id=written_id,
        action="written",
        conflict_type="NOVEL",
        chunks_written=len(chunks_dicts),
        annotations_written=len(annotations),
    )


@app.get("/v1/memory/search", response_model=SearchResponse)
async def search_memories(
    query: str,
    top_k: int = 10,
    factory: FactoryDep = None,
):
    """Search memories by semantic similarity."""
    embedding = await factory._embedder.embed(query)
    results   = await factory._vector.search(embedding, top_k=top_k)

    search_results = [
        SearchResult(
            memory_id=r.get("parent_id", ""),
            chunk_id=r.get("id", ""),
            content=r.get("content", "")[:500],
            score=round(r.get("score", 0.0), 4),
            source=r.get("source", ""),
            source_type=r.get("source_type", "prose"),
        )
        for r in results
    ]

    return SearchResponse(results=search_results, query=query, total=len(search_results))


@app.get("/v1/session/{session_id}/slots")
async def get_slots(session_id: str, factory: FactoryDep):
    """Get current staging slot state for a session."""
    session = await factory.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {"session_id": session_id, "slots": session._cache.slot_summary()}


# ── System endpoints ───────────────────────────────────────────────────────────

@app.get("/v1/stats", response_model=StatsResponse)
async def get_stats(factory: FactoryDep):
    """System-wide statistics."""
    try:
        ds_stats = await factory._dataset.get_stats()
        traj_samples = ds_stats.get("total_samples", 0)
    except Exception:
        traj_samples = 0

    return StatsResponse(
        active_sessions=len(factory._sessions),
        total_memories=0,       # TODO: query meta store
        total_chunks=0,
        trajectory_samples=traj_samples,
    )


@app.get("/v1/health", response_model=HealthResponse)
async def health_check(factory: FactoryDep):
    """Health check — verifies all store connections."""
    stores = {}
    try:
        await factory._meta._db.execute("SELECT 1")
        stores["sqlite"] = True
    except Exception:
        stores["sqlite"] = False

    try:
        await factory._vector.search(
            query_vector=__import__("numpy").zeros(factory._embedder.dim),
            top_k=1,
        )
        stores["qdrant"] = True
    except Exception:
        stores["qdrant"] = False

    all_healthy = all(stores.values())
    return HealthResponse(
        status="ok" if all_healthy else "degraded",
        stores=stores,
    )

from fastapi.responses import PlainTextResponse

@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """Prometheus metrics endpoint."""
    return metrics.generate_latest()

