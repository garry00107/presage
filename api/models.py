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

