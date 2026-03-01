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

