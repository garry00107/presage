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

