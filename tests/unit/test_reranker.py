"""Tests for core/staging/reranker.py"""

import numpy as np
import pytest
from core.staging.reranker import Reranker
from core.staging.models import StagedChunk, StagedMemory
from core.nerve.models import Prediction, PrefetchStrategy, IntentSignal
from math_core.momentum import l2_normalize


def rand_unit(d=64):
    return l2_normalize(np.random.randn(d).astype(np.float32))


def make_chunk(cid, score=0.5) -> StagedChunk:
    return StagedChunk(
        chunk_id=cid, parent_id="p1", chunk_index=0,
        content="content", tokens=50, score=score, source_type="prose",
    )


def make_staged(conf=0.8, chunk_ids=None) -> StagedMemory:
    pred = Prediction(
        query_vector=rand_unit(),
        query_text="test",
        graph_seeds=[], annotation_tags=[],
        confidence=conf,
        strategy=PrefetchStrategy.SEMANTIC,
        intent=IntentSignal.EXPLORE,
        k_steps=1,
    )
    chunks = [make_chunk(cid) for cid in (chunk_ids or ["c1"])]
    return StagedMemory(
        prediction=pred, chunks=chunks,
        raw_confidence=conf, rerank_score=0.5,
    )


reranker = Reranker()


def test_rerank_updates_scores():
    query = rand_unit()
    chunk_emb = rand_unit()
    sm = make_staged()
    sm.chunks[0].chunk_id = "c1"
    chunk_embeddings = {"c1": chunk_emb}

    result = reranker.rerank([sm], query, chunk_embeddings)
    expected_score = float(np.dot(query, chunk_emb))
    assert abs(result[0].rerank_score - expected_score) < 1e-5

def test_rerank_sorts_by_combined_score():
    query = rand_unit()

    # Make two memories with different similarities to query
    sm_high = make_staged(conf=0.9, chunk_ids=["ch"])
    sm_low  = make_staged(conf=0.5, chunk_ids=["cl"])

    # High similarity for sm_high, low for sm_low
    high_emb = query.copy()   # identical → sim=1.0
    low_emb  = l2_normalize(-query + 0.01 * np.random.randn(*query.shape))

    chunk_embeddings = {"ch": high_emb, "cl": low_emb}
    result = reranker.rerank([sm_low, sm_high], query, chunk_embeddings)

    assert result[0].combined_score >= result[1].combined_score

def test_rerank_fallback_when_no_embedding():
    """Memory without chunk embeddings keeps raw_confidence as rerank_score."""
    query = rand_unit()
    sm = make_staged(conf=0.75)
    result = reranker.rerank([sm], query, {})
    assert result[0].rerank_score == 0.75   # fallback to raw confidence

def test_rerank_chunks_updates_individual_scores():
    query = rand_unit()
    chunks = [make_chunk("c1"), make_chunk("c2")]
    c1_emb = query.copy()
    c2_emb = l2_normalize(-query)
    chunk_embeddings = {"c1": c1_emb, "c2": c2_emb}

    result = reranker.rerank_chunks(chunks, query, chunk_embeddings)
    assert result[0].chunk_id == "c1"   # higher sim should be first

def test_rerank_returns_same_count():
    query = rand_unit()
    staged = [make_staged(conf=0.8), make_staged(conf=0.6)]
    result = reranker.rerank(staged, query, {})
    assert len(result) == 2

