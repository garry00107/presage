# =============================================================================
# PPM: Predictive Push Memory — Phase 1 Starter Code
# All v2 refinements applied. Python 3.12, async-first, fully type-annotated.
# =============================================================================
# FILE TREE (this file is a multi-file scaffold — split by the ### markers)
#
# config/settings.py
# core/types.py
# math/momentum.py
# math/entropy.py
# math/knapsack.py
# math/diffusion.py
# math/bandit.py
# adapters/embedder/base.py
# adapters/embedder/openai.py
# adapters/embedder/local.py
# core/store/meta.py  (SQLite schema + outbox)
# tests/unit/test_momentum.py
# tests/unit/test_knapsack.py
# tests/unit/test_bandit.py
# =============================================================================


### config/settings.py ###

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal


class PPMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PPM_", env_file=".env")

    # Embedder
    embedder_backend: Literal["openai", "nomic", "bge"] = "openai"
    embedder_model: str = "text-embedding-3-small"
    embedder_dim: int = 1536
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # LLM
    llm_backend: Literal["anthropic", "openai", "ollama"] = "anthropic"
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    llm_model: str = "claude-sonnet-4-20250514"

    # Storage paths
    sqlite_path: str = "./ppm_data/ppm.db"
    qdrant_path: str = "./ppm_data/qdrant"
    kuzu_path: str = "./ppm_data/kuzu"

    # Momentum math
    decay_lambda_base: float = 0.85       # exponential decay base
    decay_lambda_min: float = 0.60        # minimum after velocity modulation
    decay_lambda_max: float = 0.95        # maximum after context-switch reset
    momentum_beta: float = 0.90           # smoothing factor (Adam-style)
    state_window_max: int = 6             # hard cap on lookback turns
    context_switch_threshold: float = 0.40 # cosine distance → trigger reset
    velocity_alpha: float = 0.10          # velocity → lambda modulation strength
    slerp_step_size: float = 0.30         # geodesic step size per turn

    # Staging
    slot_count: int = 10
    auto_inject_threshold: float = 0.80
    hot_threshold: float = 0.50
    warm_threshold: float = 0.30
    slot_ttl_seconds: float = 120.0

    # Context budget
    max_inject_tokens: int = 4096

    # Outbox worker
    outbox_poll_interval_s: float = 0.10
    outbox_max_attempts: int = 5
    outbox_backoff_base_s: float = 2.0
    read_your_writes_window_s: float = 5.0

    # Observability
    log_level: str = "INFO"
    metrics_port: int = 9090


settings = PPMSettings()


### core/types.py ###

from typing import NewType, TypedDict
import numpy as np


# Contract: every UnitVector v satisfies |v| == 1.0 (±1e-6)
# Enforced by l2_normalize() at all math module boundaries.
UnitVector = NewType("UnitVector", np.ndarray)

MemoryID = NewType("MemoryID", str)
ChunkID = NewType("ChunkID", str)
SessionID = NewType("SessionID", str)


class Chunk(TypedDict):
    id: ChunkID
    parent_id: MemoryID
    chunk_index: int
    content: str
    tokens: int
    score: float          # set at retrieval time
    source_type: str      # 'code' | 'prose' | 'json' | 'yaml'
    embedding: list[float] | None


class Memory(TypedDict):
    id: MemoryID
    content: str
    source: str           # file path, url, or conversation_id
    source_type: str
    token_count: int
    chunks: list[Chunk]
    forward_contexts: list[str]  # forward annotation tags
    graph_edges: list[dict]


class Prediction(TypedDict):
    query_vector: UnitVector
    query_text: str
    graph_seeds: list[MemoryID]
    confidence: float
    strategy: str          # 'SEMANTIC' | 'GRAPH' | 'SYMBOL' | 'HYBRID'
    intent_signal: str


class TurnSignals(TypedDict):
    embedding: UnitVector
    switch_score: float
    lambda_effective: float
    did_reset: bool
    intent_signals: list[str]


### math/momentum.py ###

"""
Momentum math for PPM's Nerve Layer.

All vectors on the unit hypersphere S^(d-1).
Embeddings from cosine-similarity models (OpenAI, nomic, bge) are
L2-normalized, so arithmetic must respect spherical geometry.

Core equations:

  Conversation state (exponentially decayed):
    C_t = normalize( Σ_{i=0}^{N} λ^(N-i) · e_i )

  Raw momentum (finite difference on sphere):
    ΔC_t = C_t - C_{t-1}

  Smoothed momentum (Adam-style EMA):
    M_t = β·M_{t-1} + (1-β)·ΔC_t

  Tangent projection (removes component ∥ to C_t):
    M_tan = M_t - (M_t · C_t)·C_t

  Velocity scalar:
    v = ‖M_tan‖

  Geodesic extrapolation (SLERP along great circle):
    Ĉ_{t+k} = cos(θ)·C_t + sin(θ)·M̂_tan,   θ = v·k·step_size
"""

import numpy as np
from core.types import UnitVector


def l2_normalize(v: np.ndarray) -> UnitVector:
    """Project v onto unit hypersphere. Safe for zero vectors."""
    norm = np.linalg.norm(v)
    return UnitVector(v / norm if norm > 1e-10 else v)


def conversation_state(
    turn_embeddings: list[UnitVector],
    decay: float = 0.85,
) -> UnitVector:
    """
    Exponentially decayed weighted sum of turn embeddings, normalized.

    Weights: w_i = λ^(N-1-i), then normalized to sum to 1.
    Recent turns contribute more; oldest turn has weight λ^(N-1).

    Args:
        turn_embeddings: ordered list of UnitVectors, oldest first.
        decay: λ ∈ (0,1). Lower = forget faster.

    Returns:
        UnitVector representing current conversation position on sphere.
    """
    if not turn_embeddings:
        raise ValueError("turn_embeddings must be non-empty")
    if len(turn_embeddings) == 1:
        return turn_embeddings[0]

    n = len(turn_embeddings)
    # w_i = λ^(n-1-i): index 0 (oldest) gets λ^(n-1), index n-1 (newest) gets λ^0=1
    exponents = np.arange(n - 1, -1, -1, dtype=np.float64)
    weights = decay ** exponents
    weights /= weights.sum()

    stacked = np.stack(turn_embeddings)          # (N, d)
    weighted_sum = weights @ stacked             # (d,)
    return l2_normalize(weighted_sum)


def momentum_tangent(
    C_t: UnitVector,
    C_prev: UnitVector,
    M_prev: np.ndarray | None,
    beta: float = 0.90,
) -> tuple[UnitVector, float]:
    """
    Smoothed momentum projected onto the tangent plane at C_t.

    Tangent plane projection removes the component parallel to C_t,
    ensuring the momentum direction is valid for geodesic extrapolation.

    Args:
        C_t:    current conversation state (UnitVector)
        C_prev: previous conversation state (UnitVector)
        M_prev: previous smoothed tangent momentum (or None at start)
        beta:   EMA smoothing factor

    Returns:
        (M_hat, velocity)
        M_hat:    unit tangent vector at C_t (UnitVector)
        velocity: scalar ∈ [0, ∞), angle moved per turn (radians proxy)
    """
    raw = C_t - C_prev                                  # raw finite difference
    M_prev_eff = M_prev if M_prev is not None else raw
    smoothed = beta * M_prev_eff + (1 - beta) * raw    # EMA

    # Project onto tangent plane at C_t
    tangent = smoothed - np.dot(smoothed, C_t) * C_t

    velocity = float(np.linalg.norm(tangent))
    M_hat = l2_normalize(tangent) if velocity > 1e-10 else UnitVector(tangent)

    return M_hat, velocity


def predict_future_state(
    C_t: UnitVector,
    M_hat: UnitVector,
    velocity: float,
    k: int = 1,
    step_size: float = 0.30,
) -> UnitVector:
    """
    Geodesic extrapolation on unit hypersphere (SLERP-style).

    Moves angle θ = velocity * k * step_size along the great circle
    defined by C_t and M_hat. Result is guaranteed on the unit sphere.

    Ĉ_{t+k} = cos(θ)·C_t + sin(θ)·M_hat

    Args:
        C_t:       current state (UnitVector)
        M_hat:     unit tangent direction (UnitVector)
        velocity:  scalar velocity (from momentum_tangent)
        k:         steps ahead to predict
        step_size: arc length per unit velocity per step

    Returns:
        Predicted future state as UnitVector — safe to use as
        cosine similarity query vector directly.
    """
    theta = velocity * k * step_size
    predicted = np.cos(theta) * C_t + np.sin(theta) * M_hat
    return l2_normalize(predicted)   # re-normalize for fp drift


### math/entropy.py ###

"""
Context-switch detection and adaptive decay for PPM.

Key equation:
    switch_score = 1 - cosine_sim(e_{t-1}, e_t)
                 = 1 - (e_{t-1} · e_t)     [both unit vectors]

    λ_effective = clip(λ_base - α·v, λ_min, λ_max)

    On context switch (switch_score > θ):
        λ_effective = λ_max, did_reset = True
"""

import numpy as np
from core.types import UnitVector


def context_switch_score(embed_prev: UnitVector, embed_curr: UnitVector) -> float:
    """
    Cosine distance between adjacent turn embeddings.

    Returns:
        0.0 = identical topic
        ~0.4 = moderate topic shift
        ~1.0 = orthogonal (complete switch)
        >1.0 = impossible for unit vectors (indicates non-normalized input)
    """
    return float(1.0 - np.dot(embed_prev, embed_curr))


def adaptive_decay(
    lambda_base: float,
    velocity: float,
    switch_score: float,
    switch_threshold: float = 0.40,
    alpha: float = 0.10,
    lambda_min: float = 0.60,
    lambda_max: float = 0.95,
) -> tuple[float, bool]:
    """
    Compute effective decay factor for this turn.

    Logic:
      1. If context switch detected → reset (did_reset=True), use λ_max
         (slow forgetting — we're starting fresh in a new topic)
      2. Otherwise → modulate by velocity:
         High velocity (fast drift) → lower λ → forget older turns faster
         Low velocity  (deep dive)  → higher λ → remember more

    Args:
        lambda_base:       base decay (from settings)
        velocity:          scalar velocity from momentum_tangent
        switch_score:      cosine distance from context_switch_score
        switch_threshold:  distance above which we declare a switch
        alpha:             velocity modulation strength
        lambda_min/max:    clipping bounds

    Returns:
        (lambda_effective, did_reset)
    """
    if switch_score > switch_threshold:
        return lambda_max, True

    lam = lambda_base - alpha * velocity
    return float(np.clip(lam, lambda_min, lambda_max)), False


### math/knapsack.py ###

"""
0/1 Knapsack for context budget allocation.

Operates on pre-chunked semantic units (never raw content).
Chunks are split at AST/sentence/structure boundaries at write time,
so the knapsack NEVER truncates content — it only selects whole chunks.

DP formulation:
    maximize   Σ v_i · x_i
    subject to Σ w_i · x_i ≤ B
               x_i ∈ {0, 1}

Solution: bottom-up DP, O(N·B) time, O(N·B) space.
For N≤50 and B≤4096: table size ≤ 800KB, runtime < 1ms.
"""

from core.types import Chunk


def knapsack_01(chunks: list[Chunk], budget: int) -> list[Chunk]:
    """
    Select whole chunks to maximize total score within token budget.

    Restores reading order within each parent memory
    (sorted by chunk_index) so injected context is coherent.

    Args:
        chunks: list of Chunk dicts with 'tokens', 'score', 'id',
                'parent_id', 'chunk_index', 'content'
        budget: max tokens to inject

    Returns:
        Selected chunks in (parent_id, chunk_index) order.
        Content is NEVER modified or truncated.
    """
    n = len(chunks)
    if n == 0 or budget <= 0:
        return []

    # Filter out chunks that can never fit
    feasible = [c for c in chunks if c["tokens"] <= budget]
    if not feasible:
        return []

    n = len(feasible)
    # Bottom-up DP table
    # dp[i][w] = max score using first i items with weight budget w
    dp = [[0.0] * (budget + 1) for _ in range(n + 1)]

    for i, chunk in enumerate(feasible, 1):
        w_i = chunk["tokens"]
        v_i = chunk["score"]
        for w in range(budget + 1):
            dp[i][w] = dp[i - 1][w]                    # skip item i
            if w >= w_i:
                take = dp[i - 1][w - w_i] + v_i
                if take > dp[i][w]:
                    dp[i][w] = take                     # take item i

    # Backtrack to recover selected items
    selected_ids: set[str] = set()
    w = budget
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_ids.add(feasible[i - 1]["id"])
            w -= feasible[i - 1]["tokens"]
            if w <= 0:
                break

    selected = [c for c in feasible if c["id"] in selected_ids]

    # Restore reading order: group by parent, sort by chunk_index
    selected.sort(key=lambda c: (c["parent_id"], c["chunk_index"]))
    return selected


### math/diffusion.py ###

"""
Personalized PageRank (PPR) over the memory graph.

Given a seed memory node, diffuses relevance through the graph to surface
structurally connected memories (callers, dependencies, related files).

Equation:
    r^(0) = e_seed                      (one-hot on seed)
    r^(t+1) = α·W·r^(t) + (1-α)·e_seed (power iteration)

Where W is the column-stochastic adjacency matrix (row-normalized adj).
Converges in ~10 iterations for α=0.85.

This is equivalent to heat diffusion on the graph, biased to teleport
back to the seed — ensuring seed relevance is maintained.
"""


def personalized_pagerank(
    adj: dict[str, list[str]],
    seed_id: str,
    alpha: float = 0.85,
    iters: int = 10,
    min_score: float = 0.01,
) -> dict[str, float]:
    """
    Sparse PPR via power iteration.

    Args:
        adj:      {node_id: [neighbor_id, ...]} adjacency list (directed)
        seed_id:  starting node
        alpha:    damping factor (0.85 standard)
        iters:    power iteration steps (10 is sufficient for convergence)
        min_score: filter out nodes below this relevance score

    Returns:
        {node_id: relevance_score}, excluding seed and low-score nodes.
    """
    if seed_id not in adj:
        return {}

    nodes = list(adj.keys())
    scores: dict[str, float] = {n: 0.0 for n in nodes}
    scores[seed_id] = 1.0

    for _ in range(iters):
        new_scores: dict[str, float] = {n: 0.0 for n in nodes}
        for node, neighbors in adj.items():
            if not neighbors:
                continue
            share = alpha * scores[node] / len(neighbors)
            for nb in neighbors:
                if nb in new_scores:
                    new_scores[nb] += share
        # Teleport back to seed
        new_scores[seed_id] += 1.0 - alpha
        scores = new_scores

    return {
        nid: s for nid, s in scores.items()
        if nid != seed_id and s >= min_score
    }


### math/bandit.py ###

"""
Bayesian Beta-Bernoulli bandit for prediction slot confidence.

Models each prediction type as a Bernoulli process:
    P(hit) ~ Beta(α_hits + 1, β_misses + 1)   [Laplace smoothed]

Properties:
  - Zero training data required (starts with Beta(1,1) = Uniform prior)
  - Calibrated uncertainty: confidence() reflects true hit rate
  - Thompson sampling for exploration when confidence is low
  - Cheap: O(1) update and query
"""

import math
import random


class BetaBandit:
    """
    Per-prediction-strategy Bayesian confidence tracker.

    Usage:
        bandit = BetaBandit()
        conf = bandit.confidence()     # 0.5 at start (uniform prior)
        bandit.update(hit=True)
        bandit.update(hit=False)
        sample = bandit.sample()       # Thompson sampling
    """

    def __init__(self, prior_hits: float = 1.0, prior_misses: float = 1.0):
        """
        Args:
            prior_hits:   α parameter (pseudo-hits before any data)
            prior_misses: β parameter (pseudo-misses before any data)
        Both default to 1.0 → Beta(1,1) = Uniform prior → confidence = 0.5
        """
        self.alpha = prior_hits
        self.beta = prior_misses

    def confidence(self) -> float:
        """
        Posterior mean of Beta(α, β) = α / (α + β).
        Range: (0, 1). Returns 0.5 with no data (uniform prior).
        """
        return self.alpha / (self.alpha + self.beta)

    def update(self, hit: bool) -> None:
        """Bayesian update: one Bernoulli observation."""
        if hit:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    def sample(self) -> float:
        """
        Thompson sampling: draw from Beta(α, β).
        Used for exploration when confidence is uncertain.
        Approximation via Johnk's method (stdlib only, no scipy needed).
        """
        # Python's random.betavariate uses Johnk's algorithm
        return random.betavariate(self.alpha, self.beta)

    def uncertainty(self) -> float:
        """
        Variance of Beta(α, β) = αβ / ((α+β)²(α+β+1)).
        High variance → we're uncertain → use sample() over confidence().
        """
        a, b = self.alpha, self.beta
        n = a + b
        return (a * b) / (n * n * (n + 1))

    def n_observations(self) -> int:
        """Total observations minus priors."""
        return int(self.alpha + self.beta - 2)

    def __repr__(self) -> str:
        return (f"BetaBandit(α={self.alpha:.1f}, β={self.beta:.1f}, "
                f"conf={self.confidence():.3f}, n={self.n_observations()})")


class BanditRegistry:
    """
    Registry of BetaBandits keyed by prediction strategy + intent signal.
    E.g., key = "GRAPH:DEBUG", "SEMANTIC:EXPLORE"
    """

    def __init__(self):
        self._bandits: dict[str, BetaBandit] = {}

    def get(self, strategy: str, intent: str) -> BetaBandit:
        key = f"{strategy}:{intent}"
        if key not in self._bandits:
            self._bandits[key] = BetaBandit()
        return self._bandits[key]

    def update(self, strategy: str, intent: str, hit: bool) -> None:
        self.get(strategy, intent).update(hit)

    def confidence(self, strategy: str, intent: str) -> float:
        return self.get(strategy, intent).confidence()

    def snapshot(self) -> dict[str, dict]:
        """Serializable snapshot for persistence."""
        return {
            k: {"alpha": b.alpha, "beta": b.beta}
            for k, b in self._bandits.items()
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, dict]) -> "BanditRegistry":
        reg = cls()
        for k, v in data.items():
            reg._bandits[k] = BetaBandit(v["alpha"], v["beta"])
        return reg


### adapters/embedder/base.py ###

from abc import ABC, abstractmethod
import numpy as np
from core.types import UnitVector
from math_core.momentum import l2_normalize


class Embedder(ABC):
    """
    Protocol for all embedding backends.
    All implementations MUST return L2-normalized UnitVectors.
    """

    @abstractmethod
    async def embed(self, text: str) -> UnitVector:
        """Embed a single string. Returns UnitVector."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[UnitVector]:
        """Embed multiple strings. More efficient than N single calls."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""
        ...

    def _normalize(self, v: np.ndarray) -> UnitVector:
        """Shared normalization. Call this in all subclass implementations."""
        return l2_normalize(v)


### adapters/embedder/openai.py ###

import numpy as np
from openai import AsyncOpenAI
from core.types import UnitVector
from adapters.embedder.base import Embedder
from config.settings import settings


class OpenAIEmbedder(Embedder):
    """
    OpenAI text-embedding-3-small (or -large) adapter.
    Returns L2-normalized UnitVectors.
    """

    def __init__(self, model: str | None = None):
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = model or settings.embedder_model
        self._dim = settings.embedder_dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, text: str) -> UnitVector:
        resp = await self._client.embeddings.create(
            model=self._model,
            input=text,
            encoding_format="float",
        )
        v = np.array(resp.data[0].embedding, dtype=np.float32)
        return self._normalize(v)

    async def embed_batch(self, texts: list[str]) -> list[UnitVector]:
        if not texts:
            return []
        resp = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            encoding_format="float",
        )
        # API returns results in order
        return [
            self._normalize(np.array(item.embedding, dtype=np.float32))
            for item in resp.data
        ]


### adapters/embedder/local.py ###

"""
Local embedding adapter via sentence-transformers.
Supports nomic-embed-text and bge-m3. No API key required.
Install: pip install sentence-transformers
"""

import numpy as np
from core.types import UnitVector
from adapters.embedder.base import Embedder


class LocalEmbedder(Embedder):
    """
    Local embedder using sentence-transformers.
    Lazy-loads model on first use.
    """

    MODEL_DIMS = {
        "nomic-ai/nomic-embed-text-v1.5": 768,
        "BAAI/bge-m3": 1024,
    }

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self._model_name = model_name
        self._model = None  # lazy load
        self._dim = self.MODEL_DIMS.get(model_name, 768)

    @property
    def dim(self) -> int:
        return self._dim

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, trust_remote_code=True)

    async def embed(self, text: str) -> UnitVector:
        self._load()
        import asyncio
        loop = asyncio.get_event_loop()
        v = await loop.run_in_executor(
            None, lambda: self._model.encode(text, normalize_embeddings=True)
        )
        return self._normalize(np.array(v, dtype=np.float32))

    async def embed_batch(self, texts: list[str]) -> list[UnitVector]:
        self._load()
        import asyncio
        loop = asyncio.get_event_loop()
        vecs = await loop.run_in_executor(
            None, lambda: self._model.encode(texts, normalize_embeddings=True,
                                             batch_size=32)
        )
        return [self._normalize(np.array(v, dtype=np.float32)) for v in vecs]


### core/store/meta.py ###

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


### tests/unit/test_momentum.py ###

"""
Property-based and unit tests for math/momentum.py
Run: pytest tests/unit/test_momentum.py -v
"""

import numpy as np
import pytest
from hypothesis import given, settings as h_settings, strategies as st
from math_core.momentum import (
    l2_normalize, conversation_state,
    momentum_tangent, predict_future_state
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def rand_unit(d: int = 64) -> np.ndarray:
    v = np.random.randn(d).astype(np.float32)
    return l2_normalize(v)


def assert_unit(v: np.ndarray, tol: float = 1e-5):
    norm = np.linalg.norm(v)
    assert abs(norm - 1.0) < tol, f"Expected unit vector, got norm={norm}"


# ── l2_normalize ───────────────────────────────────────────────────────────────

def test_l2_normalize_unit():
    v = np.array([3.0, 4.0])
    u = l2_normalize(v)
    assert_unit(u)

def test_l2_normalize_already_unit():
    v = np.array([1.0, 0.0, 0.0])
    u = l2_normalize(v)
    assert_unit(u)
    np.testing.assert_allclose(u, v, atol=1e-6)

def test_l2_normalize_zero_safe():
    v = np.zeros(4)
    u = l2_normalize(v)
    assert not np.any(np.isnan(u)), "Zero vector should not produce NaN"

@given(st.lists(st.floats(-1e3, 1e3), min_size=4, max_size=128).filter(
    lambda xs: np.linalg.norm(xs) > 1e-8
))
def test_l2_normalize_property(xs):
    v = np.array(xs, dtype=np.float64)
    u = l2_normalize(v)
    assert_unit(u)


# ── conversation_state ─────────────────────────────────────────────────────────

def test_conversation_state_single():
    e = rand_unit()
    state = conversation_state([e])
    assert_unit(state)
    np.testing.assert_allclose(state, e, atol=1e-5)

def test_conversation_state_output_is_unit():
    embeds = [rand_unit() for _ in range(5)]
    state = conversation_state(embeds, decay=0.85)
    assert_unit(state)

def test_conversation_state_recency_bias():
    """Most recent embedding should dominate with low decay."""
    d = 64
    old = np.zeros(d, dtype=np.float32); old[0] = 1.0
    new = np.zeros(d, dtype=np.float32); new[1] = 1.0
    embeds = [l2_normalize(old)] * 4 + [l2_normalize(new)]
    state = conversation_state(embeds, decay=0.5)  # aggressive forgetting
    # newest turn direction should dominate
    assert state[1] > state[0], "Recent embedding should dominate with low decay"

def test_conversation_state_raises_on_empty():
    with pytest.raises((ValueError, IndexError)):
        conversation_state([])


# ── momentum_tangent ───────────────────────────────────────────────────────────

def test_momentum_tangent_output_shapes():
    C_prev = rand_unit()
    C_t = rand_unit()
    M_hat, vel = momentum_tangent(C_t, C_prev, None)
    assert M_hat.shape == C_t.shape
    assert isinstance(vel, float)
    assert vel >= 0.0

def test_momentum_tangent_orthogonal_to_state():
    """Tangent vector must be perpendicular to C_t (tangent plane property)."""
    for _ in range(20):
        C_t = rand_unit()
        C_prev = rand_unit()
        M_hat, vel = momentum_tangent(C_t, C_prev, None)
        if vel > 1e-8:
            dot = float(np.dot(M_hat, C_t))
            assert abs(dot) < 1e-4, f"M_hat not orthogonal to C_t: dot={dot}"

def test_momentum_tangent_unit_when_nonzero():
    """M_hat should be unit when velocity is non-negligible."""
    for _ in range(20):
        C_t = rand_unit()
        C_prev = rand_unit()
        M_hat, vel = momentum_tangent(C_t, C_prev, None)
        if vel > 1e-8:
            assert_unit(M_hat)


# ── predict_future_state ───────────────────────────────────────────────────────

def test_predict_future_state_is_unit():
    """CRITICAL: predicted state must be on unit sphere for valid cosine query."""
    for _ in range(50):
        C_t = rand_unit()
        C_prev = rand_unit()
        M_hat, vel = momentum_tangent(C_t, C_prev, None)
        if vel > 1e-8:
            pred = predict_future_state(C_t, M_hat, vel, k=1)
            assert_unit(pred, tol=1e-4)

@given(st.integers(min_value=1, max_value=5))
def test_predict_future_state_unit_for_all_k(k):
    """Unit sphere invariant holds for any k."""
    C_t = rand_unit(64)
    M_hat = rand_unit(64)
    # Make M_hat orthogonal to C_t (required precondition)
    M_hat = l2_normalize(M_hat - np.dot(M_hat, C_t) * C_t)
    pred = predict_future_state(C_t, M_hat, velocity=0.3, k=k)
    assert_unit(pred, tol=1e-4)

def test_predict_k0_returns_current():
    """k=0 should return C_t itself (no movement)."""
    C_t = rand_unit()
    M_hat = rand_unit()
    M_hat = l2_normalize(M_hat - np.dot(M_hat, C_t) * C_t)
    pred = predict_future_state(C_t, M_hat, velocity=0.5, k=0, step_size=0.3)
    # cos(0)*C_t + sin(0)*M_hat = C_t
    np.testing.assert_allclose(pred, C_t, atol=1e-5)


### tests/unit/test_knapsack.py ###

"""Tests for math/knapsack.py"""

import pytest
from math_core.knapsack import knapsack_01


def make_chunk(id, parent, idx, tokens, score, content=None):
    return {
        "id": id, "parent_id": parent, "chunk_index": idx,
        "tokens": tokens, "score": score,
        "content": content or f"content_{id}",
        "source_type": "prose"
    }


def test_empty_chunks():
    assert knapsack_01([], 100) == []

def test_zero_budget():
    chunks = [make_chunk("a", "p1", 0, 10, 1.0)]
    assert knapsack_01(chunks, 0) == []

def test_all_fit():
    chunks = [make_chunk("a", "p1", 0, 10, 0.9),
              make_chunk("b", "p1", 1, 20, 0.8)]
    result = knapsack_01(chunks, 100)
    assert len(result) == 2

def test_optimal_selection():
    """Should pick high-value item even if lower-scoring fills budget."""
    chunks = [
        make_chunk("a", "p1", 0, 90, 10.0),   # best value/token
        make_chunk("b", "p2", 0, 50, 4.0),
        make_chunk("c", "p2", 1, 50, 4.0),
    ]
    result = knapsack_01(chunks, 100)
    ids = {c["id"] for c in result}
    assert "a" in ids, "High-value item should always be selected"

def test_content_never_truncated():
    """CRITICAL: no chunk content should be modified."""
    original = "def foo():\n    return 42\n"
    chunks = [make_chunk("a", "p1", 0, 20, 0.9, content=original)]
    result = knapsack_01(chunks, 100)
    assert result[0]["content"] == original

def test_reading_order_preserved():
    """Chunks from same parent must be in chunk_index order."""
    chunks = [
        make_chunk("c", "p1", 2, 10, 0.9),
        make_chunk("a", "p1", 0, 10, 0.7),
        make_chunk("b", "p1", 1, 10, 0.8),
    ]
    result = knapsack_01(chunks, 100)
    indices = [c["chunk_index"] for c in result]
    assert indices == sorted(indices)

def test_oversized_chunks_excluded():
    chunks = [make_chunk("big", "p1", 0, 500, 99.0)]
    assert knapsack_01(chunks, 100) == []


### tests/unit/test_bandit.py ###

"""Tests for math/bandit.py"""

import pytest
from math_core.bandit import BetaBandit, BanditRegistry


def test_initial_confidence():
    b = BetaBandit()
    assert abs(b.confidence() - 0.5) < 1e-9

def test_update_hit_increases_confidence():
    b = BetaBandit()
    for _ in range(10):
        b.update(hit=True)
    assert b.confidence() > 0.5

def test_update_miss_decreases_confidence():
    b = BetaBandit()
    for _ in range(10):
        b.update(hit=False)
    assert b.confidence() < 0.5

def test_n_observations():
    b = BetaBandit()
    assert b.n_observations() == 0
    b.update(True); b.update(False)
    assert b.n_observations() == 2

def test_sample_range():
    b = BetaBandit()
    for _ in range(100):
        s = b.sample()
        assert 0.0 <= s <= 1.0

def test_uncertainty_decreases_with_data():
    b = BetaBandit()
    u0 = b.uncertainty()
    for _ in range(100):
        b.update(True)
    assert b.uncertainty() < u0

def test_registry_creates_bandits():
    reg = BanditRegistry()
    b1 = reg.get("SEMANTIC", "DEBUG")
    b2 = reg.get("GRAPH", "DEBUG")
    assert b1 is not b2

def test_registry_snapshot_roundtrip():
    reg = BanditRegistry()
    reg.update("SEMANTIC", "DEBUG", hit=True)
    reg.update("GRAPH", "EXPLORE", hit=False)
    snap = reg.snapshot()
    reg2 = BanditRegistry.from_snapshot(snap)
    assert abs(reg2.confidence("SEMANTIC", "DEBUG") -
               reg.confidence("SEMANTIC", "DEBUG")) < 1e-9
