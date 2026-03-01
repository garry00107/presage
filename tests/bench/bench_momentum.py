"""
Benchmark suite for math_core/momentum.py

Measures latency for all math operations on the hot path.
Target: all operations < 2ms on a modern laptop CPU.

Run:
  python tests/bench/bench_momentum.py

Output:
  operation                    P50      P95      P99     mean
  l2_normalize (d=1536)       0.01ms   0.02ms   0.03ms  0.01ms
  conversation_state (N=6)    0.05ms   0.07ms   0.09ms  0.05ms
  momentum_tangent            0.02ms   0.03ms   0.04ms  0.02ms
  predict_future_state        0.01ms   0.02ms   0.02ms  0.01ms
"""

import time
import statistics
import numpy as np

# Fix import path for running directly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from math_core.momentum import (
    l2_normalize, conversation_state,
    momentum_tangent, predict_future_state,
)

N_WARMUP = 100
N_ITERS  = 10_000
DIM      = 1536   # production embedding dimension


def bench(name: str, fn, n: int = N_ITERS) -> dict:
    """Run fn n times, return latency statistics in milliseconds."""
    # Warmup
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
        "mean": round(statistics.mean(times), 3),
        "n":    n,
    }


def run_benchmarks():
    print(f"\nPPM Math Benchmarks (d={DIM}, n={N_ITERS:,})")
    print("=" * 65)
    print(f"{'Operation':<35} {'P50':>7} {'P95':>7} {'P99':>7} {'mean':>7}")
    print("-" * 65)

    results = []

    # l2_normalize
    v = np.random.randn(DIM).astype(np.float32)
    r = bench("l2_normalize", lambda: l2_normalize(v))
    results.append(r)

    # conversation_state (N=6 turns, production window)
    turns = [l2_normalize(np.random.randn(DIM).astype(np.float32)) for _ in range(6)]
    r = bench("conversation_state (N=6)", lambda: conversation_state(turns, 0.85))
    results.append(r)

    # momentum_tangent
    C_t    = l2_normalize(np.random.randn(DIM).astype(np.float32))
    C_prev = l2_normalize(np.random.randn(DIM).astype(np.float32))
    M_prev = np.random.randn(DIM).astype(np.float32) * 0.1
    r = bench("momentum_tangent", lambda: momentum_tangent(C_t, C_prev, M_prev))
    results.append(r)

    # predict_future_state
    M_hat = l2_normalize(np.random.randn(DIM).astype(np.float32))
    vel   = 0.15
    r = bench("predict_future_state (k=2)", lambda: predict_future_state(C_t, M_hat, vel, k=2))
    results.append(r)

    # Full pipeline: state → momentum → predict
    def full_pipeline():
        C = conversation_state(turns, 0.85)
        M, v = momentum_tangent(C, C_prev, M_prev)
        return predict_future_state(C, M, v, k=1)

    r = bench("full_pipeline (state+momentum+predict)", full_pipeline, n=N_ITERS // 2)
    results.append(r)

    for r in results:
        print(f"  {r['name']:<33} {r['p50']:>6}ms {r['p95']:>6}ms {r['p99']:>6}ms {r['mean']:>6}ms")

    print("-" * 65)

    # Check targets
    print("\nTarget validation (all ops < 2ms):")
    all_pass = True
    for r in results:
        status = "✓" if r["p99"] < 2.0 else "✗"
        if r["p99"] >= 2.0:
            all_pass = False
        print(f"  {status} {r['name']}: P99={r['p99']}ms")

    print(f"\n{'All targets met ✓' if all_pass else 'Some targets missed ✗'}")
    return results


if __name__ == "__main__":
    run_benchmarks()

