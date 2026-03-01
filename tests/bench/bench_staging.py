"""
Benchmark: Staging layer operations.

Measures:
  - Knapsack selection (N chunks, B budget)
  - Injector.plan() end-to-end
  - Reranker.rerank() for N staged memories
  - StagingCache slot operations

Target:
  - knapsack_01 (N=50, B=4096): < 1ms
  - injector.plan (10 memories): < 2ms
  - reranker.rerank (10 memories): < 1ms
"""

import time
import statistics
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from math_core.momentum import l2_normalize
from math_core.knapsack import knapsack_01
from core.staging.injector import Injector
from core.staging.reranker import Reranker
from core.staging.models import StagedChunk, StagedMemory
from core.nerve.models import Prediction, PrefetchStrategy, IntentSignal

N_WARMUP = 50
N_ITERS  = 5_000
DIM      = 1536


def rand_unit(d=DIM):
    return l2_normalize(np.random.randn(d).astype(np.float32))


def make_chunks(n: int, budget: int) -> list[dict]:
    return [
        {
            "id": f"chunk-{i}",
            "parent_id": f"mem-{i // 3}",
            "chunk_index": i % 3,
            "content": f"Content of chunk {i} with some realistic text here.",
            "tokens": np.random.randint(20, 150),
            "score": np.random.random(),
            "source_type": "prose",
            "source": "",
        }
        for i in range(n)
    ]


def make_staged_memories(n: int) -> list[StagedMemory]:
    mems = []
    for i in range(n):
        pred = Prediction(
            query_vector=rand_unit(), query_text="test",
            graph_seeds=[], annotation_tags=[],
            confidence=np.random.random(),
            strategy=PrefetchStrategy.SEMANTIC,
            intent=IntentSignal.EXPLORE, k_steps=1,
        )
        chunks = [
            StagedChunk(
                chunk_id=f"c-{i}-{j}", parent_id=f"m-{i}",
                chunk_index=j, content=f"chunk content {i}-{j}",
                tokens=50, score=np.random.random(), source_type="prose",
            )
            for j in range(3)
        ]
        mems.append(StagedMemory(
            prediction=pred, chunks=chunks,
            raw_confidence=np.random.random(), rerank_score=np.random.random(),
        ))
    return mems


def bench(name, fn, n=N_ITERS):
    for _ in range(N_WARMUP):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "name": name,
        "p50":  round(statistics.median(times), 3),
        "p95":  round(statistics.quantiles(times, n=20)[18], 3),
        "p99":  round(statistics.quantiles(times, n=100)[98], 3),
    }


def run_benchmarks():
    print(f"\nPPM Staging Benchmarks (n={N_ITERS:,})")
    print("=" * 60)
    print(f"{'Operation':<35} {'P50':>7} {'P95':>7} {'P99':>7}")
    print("-" * 60)

    results = []

    # Knapsack at various sizes
    for n_chunks, budget in [(10, 1000), (50, 4096), (100, 4096)]:
        chunks = make_chunks(n_chunks, budget)
        r = bench(
            f"knapsack_01 (N={n_chunks}, B={budget})",
            lambda c=chunks, b=budget: knapsack_01(c, b),
            n=N_ITERS // (n_chunks // 10),
        )
        results.append(r)

    # Injector.plan
    injector = Injector()
    staged_10 = make_staged_memories(10)
    r = bench(
        "injector.plan (10 memories)",
        lambda: injector.plan(staged_10, token_budget=4096),
        n=N_ITERS // 5,
    )
    results.append(r)

    # Reranker
    reranker = Reranker()
    staged_10 = make_staged_memories(10)
    query_emb = rand_unit()
    chunk_embs = {
        f"c-{i}-{j}": rand_unit()
        for i in range(10) for j in range(3)
    }
    r = bench(
        "reranker.rerank (10 memories)",
        lambda: reranker.rerank(staged_10, query_emb, chunk_embs),
        n=N_ITERS // 5,
    )
    results.append(r)

    for r in results:
        print(f"  {r['name']:<33} {r['p50']:>6}ms {r['p95']:>6}ms {r['p99']:>6}ms")

    targets = {"knapsack": 1.0, "injector": 2.0, "reranker": 1.0}
    print("\nTarget validation:")
    for r in results:
        key = next((k for k in targets if k in r["name"].lower()), None)
        if key:
            target = targets[key]
            ok = r["p99"] < target
            print(f"  {'✓' if ok else '✗'} {r['name']}: P99={r['p99']}ms (target <{target}ms)")

    return results


if __name__ == "__main__":
    run_benchmarks()

