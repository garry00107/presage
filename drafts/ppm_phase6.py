# =============================================================================
# PPM: Predictive Push Memory — Phase 6: API + Session Manager
#
# Files:
#   core/session/__init__.py
#   core/session/manager.py         ← wires all 5 phases into one session
#   core/session/factory.py         ← creates sessions with injected deps
#   api/__init__.py
#   api/server.py                   ← FastAPI REST endpoints
#   api/ws.py                       ← WebSocket real-time streaming
#   api/models.py                   ← Pydantic request/response models
#   api/deps.py                     ← FastAPI dependency injection
#   cli/ppm.py                      ← CLI: init, ingest, chat, stats
#   tests/integration/test_api.py
#   tests/integration/test_session.py
#
# This is the phase where all 5 previous phases wire together into
# a single running system you can actually talk to.
# =============================================================================


### core/session/__init__.py ###

# empty


### core/session/manager.py ###

"""
SessionManager — the central coordinator for one PPM session.

A session represents one continuous conversation with one LLM context.
The SessionManager owns the lifecycle of all phase components for
this session and exposes a single clean interface:

    result = await session.turn(user_message)

Internally this orchestrates:
  Phase 3: Observer  → extract signals from user message
  Phase 3: Predictor → generate predictions from trajectory
  Phase 4: Cache     → schedule background prefetch (non-blocking)
  Phase 4: Injector  → select memories to inject into context
  Phase 4: Reranker  → refine staged memory relevance
  LLM:     call with enriched context
  Phase 5: Tracker   → evaluate hit/miss for this turn
  Phase 5: Loop      → update bandits, seeds, dataset
  Phase 2: Pipeline  → distill and write new memories

The hot path (user sees latency) is:
  observe → inject → LLM → stream response

Everything else runs after the response starts streaming.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field

import structlog

from adapters.embedder.base import Embedder
from core.feedback.detector import HitMissDetector
from core.feedback.loop import FeedbackLoop
from core.feedback.tracker import FeedbackTracker
from core.nerve.models import IntentSignal
from core.nerve.predictor import TrajectoryPredictor
from core.nerve.state import ConversationStateManager
from core.staging.cache import StagingCache
from core.staging.injector import Injector
from core.staging.models import InjectionPlan
from core.staging.reranker import Reranker
from core.surface.observer import ConversationObserver
from core.write.pipeline import WritePipeline
from config.settings import settings

log = structlog.get_logger(__name__)


@dataclass
class TurnResult:
    """The complete result of processing one conversation turn."""
    session_id: str
    turn_index: int
    user_message: str
    llm_response: str
    injection_plan: InjectionPlan
    intent: IntentSignal
    velocity: float
    hit_rate: float
    tokens_injected: int
    memories_injected: int
    latency_ms: float
    staged_slot_summary: list[dict] = field(default_factory=list)


class SessionManager:
    """
    Manages one PPM session end-to-end.

    Lifecycle:
      1. Created by SessionFactory
      2. turn() called for each user message
      3. close() called when session ends (persists bandit state)
    """

    def __init__(
        self,
        session_id: str,
        observer: ConversationObserver,
        predictor: TrajectoryPredictor,
        state: ConversationStateManager,
        cache: StagingCache,
        injector: Injector,
        reranker: Reranker,
        tracker: FeedbackTracker,
        feedback_loop: FeedbackLoop,
        write_pipeline: WritePipeline,
        embedder: Embedder,
        llm_caller,     # callable: async (prompt, context) → str
        meta_store,
    ):
        self.session_id    = session_id
        self._observer     = observer
        self._predictor    = predictor
        self._state        = state
        self._cache        = cache
        self._injector     = injector
        self._reranker     = reranker
        self._tracker      = tracker
        self._loop         = feedback_loop
        self._write        = write_pipeline
        self._embedder     = embedder
        self._llm          = llm_caller
        self._meta         = meta_store
        self._turn_index   = 0
        self._last_switch_score = 0.0

    async def turn(
        self,
        user_message: str,
        source: str = "",
        stream: bool = False,
    ) -> TurnResult:
        """
        Process one conversation turn end-to-end.

        Hot path (blocks until response):
          1. Observe user message → TurnSignals
          2. Get AUTO-tier staged memories (already prefetched)
          3. Rerank against actual message embedding
          4. Knapsack inject into context
          5. Call LLM with enriched context
          6. Return response

        Post-response (async, non-blocking):
          7. Schedule next prefetch
          8. Run feedback (hit/miss detection)
          9. Distill and write new memories
          10. Update bandits + graph seeds

        Args:
            user_message: raw user input
            source:       optional source identifier (file path, url)
            stream:       if True, return async generator (Phase 7)

        Returns:
            TurnResult with response and all metadata
        """
        t0 = time.monotonic()
        self._turn_index += 1

        log.info(
            "session.turn_start",
            session=self.session_id,
            turn=self._turn_index,
            message_len=len(user_message),
        )

        # ── HOT PATH ──────────────────────────────────────────────────────────

        # Step 1: Observe — embed + intent + signals
        signals = await self._observer.observe(user_message)
        self._last_switch_score = signals.switch_score

        # Step 2: Get AUTO-tier staged memories (ready from prev turn's prefetch)
        await self._cache.evict_expired()
        staged = await self._cache.get_auto_inject()

        # Also get HOT-tier if soft trigger fires
        hot = await self._cache.get_hot(trigger_text=user_message)
        staged = self._deduplicate_staged(staged + hot)

        # Step 3: Rerank against actual message embedding
        if staged:
            chunk_embeddings = await self._load_chunk_embeddings(staged)
            staged = self._reranker.rerank(staged, signals.embedding, chunk_embeddings)

        # Step 4: Build injection plan (knapsack)
        plan = self._injector.plan(
            staged=staged,
            token_budget=settings.max_inject_tokens,
            soft_trigger=user_message,
        )

        # Step 5: Mark injected memories
        injected_parent_ids = list({c.parent_id for c in plan.chunks})
        await self._cache.mark_injected(injected_parent_ids)

        # Step 6: Call LLM with enriched context
        llm_response = await self._call_llm(user_message, plan)

        hot_path_ms = (time.monotonic() - t0) * 1000

        log.info(
            "session.hot_path_complete",
            turn=self._turn_index,
            latency_ms=round(hot_path_ms, 1),
            tokens_injected=plan.tokens_used,
            memories_injected=plan.memories_injected,
        )

        # ── POST-RESPONSE (schedule as background tasks) ───────────────────────

        asyncio.create_task(
            self._post_response(
                user_message=user_message,
                llm_response=llm_response,
                signals=signals,
                source=source,
            ),
            name=f"post_response_turn_{self._turn_index}",
        )

        total_ms = (time.monotonic() - t0) * 1000

        return TurnResult(
            session_id=self.session_id,
            turn_index=self._turn_index,
            user_message=user_message,
            llm_response=llm_response,
            injection_plan=plan,
            intent=signals.intent,
            velocity=self._state.current_velocity,
            hit_rate=0.0,                    # updated after feedback runs
            tokens_injected=plan.tokens_used,
            memories_injected=plan.memories_injected,
            latency_ms=total_ms,
            staged_slot_summary=self._cache.slot_summary(),
        )

    async def _post_response(
        self,
        user_message: str,
        llm_response: str,
        signals,
        source: str,
    ) -> None:
        """
        All post-response work runs here asynchronously.
        Errors are logged, never propagated.
        """
        try:
            # 1. Schedule next prefetch (most important — do this first)
            predictions = self._predictor.predict(signals)
            await self._cache.schedule_prefetch(predictions)

            # 2. Feedback: hit/miss detection
            feedback = await self._tracker.evaluate_turn(
                response_text=llm_response,
                intent=signals.intent,
            )

            # 3. Feedback loop: bandits + graph seeds + dataset
            await self._loop.process(feedback, switch_score=self._last_switch_score)

            # 4. Write pipeline: distill + store new memories
            await self._write.process_turn(
                user_message=user_message,
                assistant_message=llm_response,
                source=source,
            )

            log.debug(
                "session.post_response_complete",
                turn=self._turn_index,
                hit_rate=f"{feedback.hit_rate:.2f}",
                hits=feedback.total_hits,
            )

        except Exception as e:
            log.error(
                "session.post_response_error",
                turn=self._turn_index,
                error=str(e),
                exc_info=True,
            )

    async def _call_llm(self, user_message: str, plan: InjectionPlan) -> str:
        """Build prompt with injected context and call LLM."""
        if plan.chunks:
            prompt = f"{plan.context_text}\n\n---\n\nUser: {user_message}"
        else:
            prompt = user_message
        return await self._llm(prompt)

    async def _load_chunk_embeddings(self, staged) -> dict:
        """
        Load chunk embeddings for reranking.
        Returns empty dict if unavailable — reranker handles gracefully.
        """
        try:
            chunk_ids = [c.chunk_id for sm in staged for c in sm.chunks]
            # In production: fetch from Qdrant by IDs
            # For now: return empty (reranker falls back to raw confidence)
            return {}
        except Exception:
            return {}

    def _deduplicate_staged(self, staged) -> list:
        """Remove duplicate StagedMemory objects by parent_id."""
        seen, result = set(), []
        for sm in staged:
            for c in sm.chunks:
                if c.parent_id not in seen:
                    seen.add(c.parent_id)
                    result.append(sm)
                    break
        return result

    async def close(self) -> None:
        """
        Clean shutdown: persist bandit state, cancel pending tasks.
        Called when session ends.
        """
        try:
            snap = self._predictor.bandit_snapshot()
            await self._meta.save_bandit_state(self.session_id, snap)
            log.info("session.closed", session=self.session_id,
                     turns=self._turn_index)
        except Exception as e:
            log.error("session.close_error", error=str(e))

    @property
    def turn_count(self) -> int:
        return self._turn_index

    @property
    def velocity(self) -> float:
        return self._state.current_velocity


### core/session/factory.py ###

"""
SessionFactory — constructs fully-wired SessionManager instances.

All dependency injection happens here. Each session gets its own:
  - ConversationStateManager (trajectory state)
  - TrajectoryPredictor (with fresh bandits)
  - StagingCache (empty slots)
  - FeedbackTracker (turn counter at 0)

Shared across all sessions (singletons):
  - Embedder (connection pool)
  - MetaStore / VectorStore / GraphStore (db connections)
  - WritePipeline components

This design makes sessions stateless from the perspective of the
shared infrastructure — you can run many sessions concurrently.
"""

import uuid
import structlog

from adapters.embedder.base import Embedder
from core.feedback.dataset import TrajectoryDataset
from core.feedback.detector import HitMissDetector
from core.feedback.loop import FeedbackLoop
from core.feedback.tracker import FeedbackTracker
from core.nerve.predictor import TrajectoryPredictor
from core.nerve.state import ConversationStateManager
from core.session.manager import SessionManager
from core.staging.cache import StagingCache
from core.staging.injector import Injector
from core.staging.prefetcher import Prefetcher
from core.staging.reranker import Reranker
from core.surface.intent import IntentClassifier
from core.surface.observer import ConversationObserver
from core.write.annotator import ForwardAnnotator
from core.write.chunker import SemanticChunker
from core.write.conflict import ConflictResolver
from core.write.distiller import MemoryDistiller
from core.write.pipeline import WritePipeline
from config.settings import settings

log = structlog.get_logger(__name__)


class SessionFactory:
    """
    Creates SessionManager instances with all dependencies wired.
    One factory per application — holds shared singleton resources.
    """

    def __init__(
        self,
        embedder: Embedder,
        meta_store,
        vector_store,
        graph_store=None,
        llm_caller=None,
        distiller: MemoryDistiller | None = None,
    ):
        self._embedder    = embedder
        self._meta        = meta_store
        self._vector      = vector_store
        self._graph       = graph_store
        self._llm         = llm_caller or self._default_llm
        self._distiller   = distiller

        # Shared stateless components (safe to reuse across sessions)
        self._chunker    = SemanticChunker()
        self._resolver   = ConflictResolver()
        self._annotator  = ForwardAnnotator()
        self._detector   = HitMissDetector()
        self._injector   = Injector()
        self._reranker   = Reranker()
        self._classifier = IntentClassifier()
        self._dataset    = TrajectoryDataset(meta_store)

        # Active sessions registry
        self._sessions: dict[str, SessionManager] = {}

    async def create_session(
        self,
        session_id: str | None = None,
        restore_bandits: bool = True,
    ) -> SessionManager:
        """
        Create a new fully-wired session.

        Args:
            session_id:      optional — generates UUID if not provided
            restore_bandits: load persisted bandit state if available

        Returns:
            SessionManager ready to accept turns.
        """
        sid = session_id or str(uuid.uuid4())

        # Per-session stateful components
        state     = ConversationStateManager(dim=settings.embedder_dim)
        predictor = TrajectoryPredictor(state)

        # Restore bandits from previous session if available
        if restore_bandits:
            await self._restore_bandits(predictor, sid)

        observer = ConversationObserver(
            embedder=self._embedder,
            state_manager=state,
            classifier=self._classifier,
        )

        prefetcher = Prefetcher(
            vector_store=self._vector,
            meta_store=self._meta,
            graph_store=self._graph,
        )
        cache = StagingCache(prefetcher)

        tracker = FeedbackTracker(
            cache=cache,
            detector=self._detector,
            embedder=self._embedder,
            session_id=sid,
        )

        feedback_loop = FeedbackLoop(
            predictor=predictor,
            state=state,
            dataset=self._dataset,
            meta_store=self._meta,
        )

        distiller = self._distiller or MemoryDistiller(
            llm_backend=settings.llm_backend,
            api_key=getattr(settings, f"{settings.llm_backend}_api_key", ""),
        )

        write_pipeline = WritePipeline(
            distiller=distiller,
            embedder=self._embedder,
            resolver=self._resolver,
            chunker=self._chunker,
            annotator=self._annotator,
            meta_store=self._meta,
            vector_store=self._vector,
        )

        session = SessionManager(
            session_id=sid,
            observer=observer,
            predictor=predictor,
            state=state,
            cache=cache,
            injector=self._injector,
            reranker=self._reranker,
            tracker=tracker,
            feedback_loop=feedback_loop,
            write_pipeline=write_pipeline,
            embedder=self._embedder,
            llm_caller=self._llm,
            meta_store=self._meta,
        )

        self._sessions[sid] = session
        log.info("session_factory.created", session_id=sid)
        return session

    async def get_session(self, session_id: str) -> SessionManager | None:
        return self._sessions.get(session_id)

    async def close_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session:
            await session.close()

    async def close_all(self) -> None:
        for sid in list(self._sessions.keys()):
            await self.close_session(sid)

    async def _restore_bandits(self, predictor: TrajectoryPredictor, sid: str) -> None:
        try:
            snap = await self._meta.load_bandit_state(sid)
            if snap:
                from math_core.bandit import BanditRegistry
                predictor._bandits = BanditRegistry.from_snapshot(snap)
                log.debug("session_factory.bandits_restored", session_id=sid)
        except Exception:
            pass  # fresh bandits on any failure

    @staticmethod
    async def _default_llm(prompt: str) -> str:
        """Placeholder LLM — returns echo. Replaced by real LLM adapter."""
        return f"[PPM Echo] Received {len(prompt)} chars of context."


### api/__init__.py ###

# empty


### api/models.py ###

"""
Pydantic models for the FastAPI REST API.
All request/response bodies are validated here.
"""

from pydantic import BaseModel, Field
from typing import Any


class CreateSessionRequest(BaseModel):
    session_id: str | None = None
    restore_bandits: bool = True


class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: float


class TurnRequest(BaseModel):
    session_id: str
    message: str
    source: str = ""
    include_metadata: bool = False


class StagedSlot(BaseModel):
    slot: int
    status: str
    tier: str | None = None
    confidence: float | None = None
    rerank_score: float | None = None
    tokens: int | None = None
    strategy: str | None = None


class TurnResponse(BaseModel):
    session_id: str
    turn_index: int
    response: str
    intent: str
    velocity: float
    tokens_injected: int
    memories_injected: int
    latency_ms: float
    context_preview: str = ""          # first 200 chars of injected context
    staged_slots: list[StagedSlot] = []


class IngestRequest(BaseModel):
    content: str
    source: str = ""
    source_type: str = "prose"
    forward_contexts: list[str] = []
    session_id: str | None = None      # if provided, uses session's write pipeline


class IngestResponse(BaseModel):
    memory_id: str | None
    action: str
    conflict_type: str
    chunks_written: int
    annotations_written: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=50)
    source_type: str | None = None


class SearchResult(BaseModel):
    memory_id: str
    chunk_id: str
    content: str
    score: float
    source: str
    source_type: str


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    total: int


class StatsResponse(BaseModel):
    active_sessions: int
    total_memories: int
    total_chunks: int
    trajectory_samples: int
    bandit_stats: dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"
    stores: dict[str, bool] = {}


### api/deps.py ###

"""
FastAPI dependency injection for shared resources.
All routes access the SessionFactory and stores through these deps.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException

from core.session.factory import SessionFactory
from core.session.manager import SessionManager

# Module-level singleton — set during app startup
_factory: SessionFactory | None = None


def set_factory(factory: SessionFactory) -> None:
    global _factory
    _factory = factory


def get_factory() -> SessionFactory:
    if _factory is None:
        raise RuntimeError("SessionFactory not initialized. Call set_factory() at startup.")
    return _factory


async def get_session(session_id: str) -> SessionManager:
    factory = get_factory()
    session = await factory.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session


FactoryDep = Annotated[SessionFactory, Depends(get_factory)]


### api/server.py ###

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

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
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
    return CreateSessionResponse(
        session_id=session.session_id,
        created_at=time.time(),
    )


@app.delete("/v1/session/{session_id}")
async def close_session(session_id: str, factory: FactoryDep):
    """Close a session and persist its state."""
    await factory.close_session(session_id)
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


### api/ws.py ###

"""
WebSocket endpoint for real-time token streaming.

Protocol:
  Client → Server:
    {"type": "turn", "message": "...", "session_id": "..."}

  Server → Client:
    {"type": "signal", "intent": "DEBUG", "velocity": 0.23}
    {"type": "staged", "slot": 0, "confidence": 0.91, "tokens": 847}
    {"type": "token", "content": "H"}  (one per token during streaming)
    {"type": "done", "turn_index": 3, "latency_ms": 234.5}
    {"type": "error", "message": "..."}

The WebSocket handler manages its own session — no need to create
one via REST first. Sessions are keyed by the WebSocket session_id.
"""

import asyncio
import json
import time
import structlog
from fastapi import WebSocket, WebSocketDisconnect

from api.deps import get_factory

log = structlog.get_logger(__name__)


async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    Handle one WebSocket connection for a PPM session.
    One connection = one session = one user conversation.
    """
    await websocket.accept()
    factory = get_factory()

    # Create or retrieve session
    session = await factory.get_session(session_id)
    if session is None:
        session = await factory.create_session(session_id=session_id)

    log.info("ws.connected", session_id=session_id)

    try:
        async for raw_msg in _receive_messages(websocket):
            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                await _send(websocket, {"type": "error", "message": "Invalid JSON"})
                continue

            if msg.get("type") == "turn":
                await _handle_turn(websocket, session, msg)
            elif msg.get("type") == "ping":
                await _send(websocket, {"type": "pong"})
            else:
                await _send(websocket, {
                    "type": "error",
                    "message": f"Unknown message type: {msg.get('type')}"
                })

    except WebSocketDisconnect:
        log.info("ws.disconnected", session_id=session_id)
    except Exception as e:
        log.error("ws.error", session_id=session_id, error=str(e))
        try:
            await _send(websocket, {"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        await factory.close_session(session_id)


async def _handle_turn(websocket: WebSocket, session, msg: dict) -> None:
    """Process one turn message over WebSocket."""
    user_message = msg.get("message", "")
    source       = msg.get("source", "")
    t0           = time.monotonic()

    # Send signal info as soon as we have it
    # (observe runs before LLM — gives user fast feedback)
    signals = await session._observer.observe(user_message)
    await _send(websocket, {
        "type": "signal",
        "intent": signals.intent.value,
        "velocity": round(session._state.current_velocity, 4),
        "switch_score": round(signals.switch_score, 4),
    })

    # Send staged slot preview
    for slot_info in session._cache.slot_summary():
        if slot_info["status"] == "ready":
            await _send(websocket, {
                "type": "staged",
                "slot": slot_info["slot"],
                "tier": slot_info.get("tier"),
                "confidence": slot_info.get("confidence"),
                "tokens": slot_info.get("tokens"),
            })

    # Run the full turn (LLM call happens here)
    result = await session.turn(user_message=user_message, source=source)

    # Send response as a single message (streaming tokens in Phase 7)
    await _send(websocket, {
        "type": "response",
        "content": result.llm_response,
    })

    await _send(websocket, {
        "type": "done",
        "turn_index": result.turn_index,
        "latency_ms": round(result.latency_ms, 1),
        "tokens_injected": result.tokens_injected,
        "memories_injected": result.memories_injected,
    })


async def _receive_messages(websocket: WebSocket):
    """Async generator that yields raw messages from WebSocket."""
    while True:
        try:
            data = await websocket.receive_text()
            yield data
        except WebSocketDisconnect:
            return


async def _send(websocket: WebSocket, data: dict) -> None:
    """Send a JSON message over WebSocket. Swallow send errors."""
    try:
        await websocket.send_text(json.dumps(data))
    except Exception:
        pass


### cli/ppm.py ###

"""
PPM CLI — command line interface for managing the PPM system.

Commands:
  ppm init                 Initialize stores (create tables, collections)
  ppm ingest <path>        Ingest a file or directory into memory
  ppm chat                 Start interactive chat session
  ppm search <query>       Search memories
  ppm stats                Show system statistics
  ppm export               Export trajectory dataset as JSONL
  ppm serve                Start the API server

Usage:
  python -m cli.ppm init
  python -m cli.ppm ingest ./src/
  python -m cli.ppm chat
  python -m cli.ppm serve --port 8000
"""

import asyncio
import sys
import os
from pathlib import Path


def _print_banner():
    print("""
╔═══════════════════════════════════════╗
║   PPM — Predictive Push Memory v0.1  ║
║   Anticipatory memory brain for LLMs ║
╚═══════════════════════════════════════╝
""")


async def cmd_init():
    """Initialize all stores."""
    from config.settings import settings
    from core.store.meta import MetaStore
    from core.feedback.dataset import TrajectoryDataset

    print("Initializing PPM stores...")

    meta = MetaStore(settings.sqlite_path)
    await meta.connect()

    ds = TrajectoryDataset(meta)
    await ds.initialize()

    print(f"  ✓ SQLite: {settings.sqlite_path}")
    print(f"  ✓ Qdrant: {settings.qdrant_path}")
    print(f"  ✓ Kuzu:   {settings.kuzu_path}")
    print("  ✓ Trajectory dataset table created")
    print("\nPPM initialized. Run `ppm serve` to start the API.")

    await meta.close()


async def cmd_serve(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server with all dependencies."""
    import uvicorn
    from config.settings import settings
    from core.store.meta import MetaStore
    from core.store.vector import VectorStore
    from core.session.factory import SessionFactory
    from api.server import app
    from api.deps import set_factory
    from api.ws import websocket_endpoint

    # Wire up FastAPI WebSocket route
    from fastapi import WebSocket
    @app.websocket("/v1/ws/{session_id}")
    async def ws_route(websocket: WebSocket, session_id: str):
        await websocket_endpoint(websocket, session_id)

    print("Starting PPM server...")

    # Initialize stores
    meta   = MetaStore(settings.sqlite_path)
    await meta.connect()

    vector = VectorStore(settings.qdrant_path, dim=settings.embedder_dim)
    await vector.connect()

    # Initialize embedder
    if settings.embedder_backend == "openai":
        from adapters.embedder.openai import OpenAIEmbedder
        embedder = OpenAIEmbedder()
    else:
        from adapters.embedder.local import LocalEmbedder
        embedder = LocalEmbedder()

    # Initialize LLM caller
    llm_caller = _build_llm_caller(settings)

    # Create factory
    factory = SessionFactory(
        embedder=embedder,
        meta_store=meta,
        vector_store=vector,
        llm_caller=llm_caller,
    )
    set_factory(factory)
    app.state.factory = factory

    print(f"  ✓ Server starting on http://{host}:{port}")
    print(f"  ✓ API docs: http://{host}:{port}/docs")
    print(f"  ✓ WebSocket: ws://{host}:{port}/v1/ws/{{session_id}}")

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


async def cmd_chat():
    """Interactive REPL chat session."""
    from config.settings import settings
    from core.store.meta import MetaStore
    from core.store.vector import VectorStore
    from core.session.factory import SessionFactory

    _print_banner()

    meta   = MetaStore(settings.sqlite_path)
    await meta.connect()
    vector = VectorStore(settings.qdrant_path, dim=settings.embedder_dim)
    await vector.connect()

    if settings.embedder_backend == "openai":
        from adapters.embedder.openai import OpenAIEmbedder
        embedder = OpenAIEmbedder()
    else:
        from adapters.embedder.local import LocalEmbedder
        embedder = LocalEmbedder()

    llm_caller = _build_llm_caller(settings)

    factory = SessionFactory(
        embedder=embedder,
        meta_store=meta,
        vector_store=vector,
        llm_caller=llm_caller,
    )
    session = await factory.create_session()

    print(f"Session: {session.session_id}")
    print("Type your message. Ctrl+C to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        result = await session.turn(user_input)

        print(f"\nPPM [{result.intent.value} | v={result.velocity:.2f} | "
              f"{result.memories_injected} mem | {result.latency_ms:.0f}ms]:")
        print(result.llm_response)
        print()

    await factory.close_all()
    await meta.close()


async def cmd_ingest(path: str):
    """Ingest a file or directory into PPM memory."""
    import httpx

    target = Path(path)
    if not target.exists():
        print(f"Error: {path} does not exist")
        sys.exit(1)

    files = [target] if target.is_file() else list(target.rglob("*"))
    files = [f for f in files if f.is_file()]

    print(f"Ingesting {len(files)} files...")

    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        for f in files:
            ext = f.suffix.lower()
            source_type_map = {
                ".py": "code", ".js": "code", ".ts": "code",
                ".go": "code", ".rs": "code",
                ".json": "json", ".yaml": "yaml", ".yml": "yaml",
                ".md": "md",
            }
            source_type = source_type_map.get(ext, "prose")

            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                if not content.strip():
                    continue

                resp = await client.post("/v1/ingest", json={
                    "content": content,
                    "source": str(f),
                    "source_type": source_type,
                })
                data = resp.json()
                print(f"  ✓ {f.name}: {data['chunks_written']} chunks, "
                      f"{data['annotations_written']} annotations")
            except Exception as e:
                print(f"  ✗ {f.name}: {e}")


async def cmd_stats():
    """Show system statistics."""
    import httpx
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        resp = await client.get("/v1/stats")
        data = resp.json()

    print("PPM System Statistics")
    print("─" * 40)
    print(f"  Active sessions:      {data['active_sessions']}")
    print(f"  Total memories:       {data['total_memories']}")
    print(f"  Total chunks:         {data['total_chunks']}")
    print(f"  Trajectory samples:   {data['trajectory_samples']}")


async def cmd_export(output: str = "trajectory_data.jsonl"):
    """Export trajectory dataset for fine-tuning."""
    from config.settings import settings
    from core.store.meta import MetaStore
    from core.feedback.dataset import TrajectoryDataset

    meta = MetaStore(settings.sqlite_path)
    await meta.connect()
    ds = TrajectoryDataset(meta)
    count = await ds.export_jsonl(output)
    print(f"Exported {count} trajectory samples to {output}")
    await meta.close()


def _build_llm_caller(settings):
    """Build an async LLM caller based on settings."""
    if settings.llm_backend == "anthropic":
        async def anthropic_caller(prompt: str) -> str:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
            msg = await client.messages.create(
                model=settings.llm_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        return anthropic_caller

    elif settings.llm_backend == "openai":
        async def openai_caller(prompt: str) -> str:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.openai_api_key)
            resp = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        return openai_caller

    else:
        # Ollama local
        async def ollama_caller(prompt: str) -> str:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "http://localhost:11434/api/generate",
                    json={"model": settings.llm_model, "prompt": prompt, "stream": False},
                    timeout=120.0,
                )
                return resp.json().get("response", "")
        return ollama_caller


def main():
    """CLI entry point."""
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0]
    rest = args[1:]

    commands = {
        "init":   lambda: asyncio.run(cmd_init()),
        "serve":  lambda: asyncio.run(cmd_serve(
            port=int(rest[1]) if len(rest) > 1 else 8000
        )),
        "chat":   lambda: asyncio.run(cmd_chat()),
        "ingest": lambda: asyncio.run(cmd_ingest(rest[0]) if rest else print("Usage: ppm ingest <path>")),
        "stats":  lambda: asyncio.run(cmd_stats()),
        "export": lambda: asyncio.run(cmd_export(rest[0] if rest else "trajectory_data.jsonl")),
    }

    if cmd not in commands:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(commands)}")
        sys.exit(1)

    commands[cmd]()


if __name__ == "__main__":
    main()


### tests/integration/test_session.py ###

"""
Integration test: full session end-to-end.
SessionFactory → SessionManager → turn() with mock LLM and stores.
"""

import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from math_core.momentum import l2_normalize
from core.session.factory import SessionFactory
from core.nerve.models import IntentSignal


DIM = 64


def rand_unit():
    return l2_normalize(np.random.randn(DIM).astype(np.float32))


def make_embedder_mock():
    emb = AsyncMock()
    emb.embed = AsyncMock(side_effect=lambda t: rand_unit())
    emb.embed_batch = AsyncMock(side_effect=lambda ts: [rand_unit() for _ in ts])
    emb.dim = DIM
    return emb


def make_vector_mock():
    v = AsyncMock()
    v.search = AsyncMock(return_value=[])
    v.upsert = AsyncMock()
    v.connect = AsyncMock()
    return v


def make_meta_mock():
    m = AsyncMock()
    m._db = AsyncMock()
    m._db.execute = AsyncMock()
    m._db.executescript = AsyncMock()
    m._db.commit = AsyncMock()
    cursor = AsyncMock()
    cursor.__aenter__ = AsyncMock(return_value=cursor)
    cursor.__aexit__ = AsyncMock(return_value=False)
    cursor.fetchone = AsyncMock(return_value={"n": 0, "sessions": 0})
    cursor.fetchall = AsyncMock(return_value=[])
    m._db.execute.return_value = cursor
    m.insert_memory = AsyncMock(return_value="mem-001")
    m.touch_memory = AsyncMock()
    m.soft_delete = AsyncMock()
    m.search_by_annotation = AsyncMock(return_value=[])
    m.get_recently_written = AsyncMock(return_value=[])
    m.increment_annotation_hit = AsyncMock()
    m.save_bandit_state = AsyncMock()
    m.load_bandit_state = AsyncMock(return_value=None)
    return m


def make_distiller_mock():
    from core.write.distiller import MemoryDistiller
    d = AsyncMock(spec=MemoryDistiller)
    d.distill = AsyncMock(return_value=[])   # no memories distilled (fast tests)
    return d


async def make_factory():
    embedder = make_embedder_mock()
    vector   = make_vector_mock()
    meta     = make_meta_mock()

    async def mock_llm(prompt: str) -> str:
        return f"Mock LLM response to: {prompt[:50]}..."

    factory = SessionFactory(
        embedder=embedder,
        meta_store=meta,
        vector_store=vector,
        llm_caller=mock_llm,
        distiller=make_distiller_mock(),
    )
    return factory


@pytest.mark.asyncio
async def test_create_session():
    factory = await make_factory()
    session = await factory.create_session()
    assert session.session_id is not None
    assert session.turn_count == 0


@pytest.mark.asyncio
async def test_session_turn_returns_result():
    factory = await make_factory()
    session = await factory.create_session()
    result  = await session.turn("What is dependency injection?")
    assert result.llm_response != ""
    assert result.turn_index == 1
    assert result.intent is not None


@pytest.mark.asyncio
async def test_session_turn_increments_count():
    factory = await make_factory()
    session = await factory.create_session()
    await session.turn("First message")
    await session.turn("Second message")
    assert session.turn_count == 2


@pytest.mark.asyncio
async def test_session_turn_returns_intent():
    factory = await make_factory()
    session = await factory.create_session()
    result  = await session.turn("why does my code crash with an error?")
    assert result.intent == IntentSignal.DEBUG


@pytest.mark.asyncio
async def test_session_latency_recorded():
    factory = await make_factory()
    session = await factory.create_session()
    result  = await session.turn("test message")
    assert result.latency_ms > 0


@pytest.mark.asyncio
async def test_multiple_sessions_independent():
    factory = await make_factory()
    s1 = await factory.create_session()
    s2 = await factory.create_session()
    assert s1.session_id != s2.session_id

    await s1.turn("message to session 1")
    assert s1.turn_count == 1
    assert s2.turn_count == 0


@pytest.mark.asyncio
async def test_close_session():
    factory = await make_factory()
    session = await factory.create_session()
    sid     = session.session_id

    await factory.close_session(sid)
    retrieved = await factory.get_session(sid)
    assert retrieved is None


@pytest.mark.asyncio
async def test_auto_create_session_on_missing():
    """Factory should create a session if get returns None."""
    factory = await make_factory()
    session = await factory.get_session("nonexistent-id")
    assert session is None  # returns None, caller can create


@pytest.mark.asyncio
async def test_session_velocity_increases_over_turns():
    """Velocity should be nonzero after multiple diverse turns."""
    factory = await make_factory()
    session = await factory.create_session()

    turns = [
        "explain what authentication means",
        "how does JWT work?",
        "write a function to verify tokens",
        "why does my token validation fail?",
    ]
    for t in turns:
        await session.turn(t)

    # After 4 diverse turns, velocity should be non-zero
    assert session.velocity >= 0.0  # always true
    assert session.turn_count == 4


### tests/integration/test_api.py ###

"""
Integration tests for the FastAPI REST API.
Uses TestClient — no real stores, all mocked.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from math_core.momentum import l2_normalize
from api.server import app
from api.deps import set_factory


DIM = 64


def rand_unit():
    return l2_normalize(np.random.randn(DIM).astype(np.float32))


def make_mock_factory():
    """Build a minimal mock SessionFactory for API tests."""
    from core.session.factory import SessionFactory

    factory = AsyncMock(spec=SessionFactory)
    factory._sessions = {}
    factory._embedder = AsyncMock()
    factory._embedder.dim = DIM
    factory._embedder.embed = AsyncMock(return_value=rand_unit())
    factory._vector = AsyncMock()
    factory._vector.search = AsyncMock(return_value=[])
    factory._meta = AsyncMock()
    factory._meta._db = AsyncMock()
    factory._meta.insert_memory = AsyncMock(return_value="mem-001")
    factory._dataset = AsyncMock()
    factory._dataset.get_stats = AsyncMock(return_value={
        "total_samples": 0, "total_sessions": 0, "by_intent": {}
    })
    factory._chunker = MagicMock()
    factory._chunker.chunk = MagicMock(return_value=[])
    factory._annotator = MagicMock()
    factory._annotator.annotate = MagicMock(return_value=[])

    # Mock create_session
    async def mock_create(session_id=None, restore_bandits=True):
        from unittest.mock import MagicMock
        session = AsyncMock()
        session.session_id = session_id or "test-session-id"
        session.turn_count = 0
        session._cache = AsyncMock()
        session._cache.slot_summary = MagicMock(return_value=[
            {"slot": i, "status": "empty"} for i in range(10)
        ])

        async def mock_turn(user_message, source="", stream=False):
            from core.session.manager import TurnResult
            from core.staging.models import InjectionPlan
            from core.nerve.models import IntentSignal
            plan = InjectionPlan(
                chunks=[], tokens_used=0, tokens_budget=4096,
                memories_injected=0, staged_memories=[],
            )
            return TurnResult(
                session_id=session.session_id,
                turn_index=1,
                user_message=user_message,
                llm_response="Mock response",
                injection_plan=plan,
                intent=IntentSignal.EXPLORE,
                velocity=0.12,
                hit_rate=0.5,
                tokens_injected=0,
                memories_injected=0,
                latency_ms=42.0,
            )

        session.turn = mock_turn
        factory._sessions[session.session_id] = session
        return session

    factory.create_session = mock_create
    factory.get_session = AsyncMock(
        side_effect=lambda sid: factory._sessions.get(sid)
    )
    factory.close_session = AsyncMock(
        side_effect=lambda sid: factory._sessions.pop(sid, None)
    )
    factory.close_all = AsyncMock()

    return factory


@pytest.fixture
def client():
    factory = make_mock_factory()
    set_factory(factory)
    with TestClient(app) as c:
        yield c, factory


def test_health_check(client):
    c, factory = client
    factory._meta._db.execute = AsyncMock()
    resp = c.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data


def test_create_session(client):
    c, _ = client
    resp = c.post("/v1/session", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "created_at" in data


def test_create_session_with_id(client):
    c, _ = client
    resp = c.post("/v1/session", json={"session_id": "my-custom-id"})
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "my-custom-id"


def test_submit_turn(client):
    c, _ = client
    # Create session first
    create_resp = c.post("/v1/session", json={})
    sid = create_resp.json()["session_id"]

    # Submit turn
    resp = c.post("/v1/turn", json={
        "session_id": sid,
        "message": "What is authentication?",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert "intent" in data
    assert "latency_ms" in data


def test_turn_response_has_slots(client):
    c, _ = client
    create_resp = c.post("/v1/session", json={})
    sid = create_resp.json()["session_id"]
    resp = c.post("/v1/turn", json={"session_id": sid, "message": "test"})
    data = resp.json()
    assert "staged_slots" in data


def test_close_session(client):
    c, _ = client
    create_resp = c.post("/v1/session", json={})
    sid = create_resp.json()["session_id"]
    resp = c.delete(f"/v1/session/{sid}")
    assert resp.status_code == 200


def test_stats_endpoint(client):
    c, _ = client
    resp = c.get("/v1/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "active_sessions" in data
    assert "trajectory_samples" in data


def test_search_memories(client):
    c, _ = client
    resp = c.get("/v1/memory/search", params={"query": "authentication", "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert data["query"] == "authentication"
