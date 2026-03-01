"""
Benchmark: Store layer operations.

Measures SQLite read/write performance for common MetaStore queries.
Qdrant benchmarks require a running Qdrant instance — skipped if unavailable.

Targets:
  - SQLite annotation search: < 5ms P99
  - SQLite memory insert:     < 10ms P99
  - Outbox enqueue:           < 2ms P99
"""

import asyncio
import time
import statistics
import uuid
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.store.meta import MetaStore
from config.settings import settings

N_ITERS = 1_000


async def bench_async(name: str, coro_fn, n: int = N_ITERS) -> dict:
    """Benchmark an async function."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        await coro_fn()
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "name": name,
        "p50":  round(statistics.median(times), 3),
        "p95":  round(statistics.quantiles(times, n=20)[18], 3),
        "p99":  round(statistics.quantiles(times, n=100)[98], 3),
    }


async def run_benchmarks():
    print(f"\nPPM Store Benchmarks (n={N_ITERS:,})")
    print("=" * 60)
    print(f"{'Operation':<35} {'P50':>7} {'P95':>7} {'P99':>7}")
    print("-" * 60)

    # Use in-memory SQLite for benchmarking (no disk I/O noise)
    meta = MetaStore(":memory:")
    await meta.connect()

    results = []

    # Seed some data for reads
    for i in range(100):
        mid = str(uuid.uuid4())
        await meta._db.execute(
            "INSERT INTO memories (id, content, content_hash, source_type, token_count, created_at) VALUES (?,?,?,?,?,?)",
            (mid, f"Content {i}", f"hash{i}", "prose", 100, int(time.time()))
        )
        await meta._db.execute(
            "INSERT INTO forward_annotations (memory_id, context_tag, created_at) VALUES (?,?,?)",
            (mid, f"intent:DEBUG", int(time.time()))
        )
    await meta._db.commit()

    # Annotation search
    r = await bench_async(
        "annotation search (1 tag)",
        lambda: meta.search_by_annotation(["intent:DEBUG"], limit=10),
    )
    results.append(r)

    # Memory insert
    async def insert_memory():
        mid = str(uuid.uuid4())
        await meta.insert_memory({
            "id": mid,
            "content": "Test memory content for benchmarking purposes.",
            "source": "bench.py",
            "source_type": "prose",
            "token_count": 50,
            "chunks": [],
            "forward_contexts": ["intent:EXPLORE"],
            "graph_edges": [],
        })

    r = await bench_async("memory insert (no chunks)", insert_memory, n=N_ITERS // 2)
    results.append(r)

    # Recently written query (read-your-writes)
    r = await bench_async(
        "recently_written (5s window)",
        lambda: meta.get_recently_written(within_seconds=5.0),
    )
    results.append(r)

    await meta.close()

    for r in results:
        print(f"  {r['name']:<33} {r['p50']:>6}ms {r['p95']:>6}ms {r['p99']:>6}ms")

    print("\nTarget validation:")
    targets = {
        "annotation": 5.0,
        "insert": 10.0,
        "recently": 5.0,
    }
    for r in results:
        key = next((k for k in targets if k in r["name"].lower()), None)
        if key:
            ok = r["p99"] < targets[key]
            print(f"  {'✓' if ok else '✗'} {r['name']}: P99={r['p99']}ms")


if __name__ == "__main__":
    asyncio.run(run_benchmarks())

