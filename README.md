# NOTE: Save as README.md in project root (no .py extension)

README_CONTENT = """
# PPM — Predictive Push Memory

> An anticipatory memory brain for LLMs. Retrieves before you ask.

PPM is a novel memory architecture that **pushes** relevant context to the LLM
proactively — before the user's next message arrives — rather than reactively
retrieving it after the fact.

## Architecture

```
User Message
    │
    ├──[HOT PATH]────────────────────────────────────────────────────┐
    │  Observer: embed + intent + signals (< 2ms)                    │
    │  Staging Cache: grab pre-fetched AUTO slots (< 1ms)            │
    │  Reranker: refine against actual message (< 1ms)               │
    │  Knapsack Injector: fit into context budget (< 1ms)            │
    │  LLM: call with enriched context                               │
    └──[BACKGROUND]──────────────────────────────────────────────────┘
       Predictor: geodesic extrapolation → next predictions
       Prefetcher: fetch from vector + graph + annotation stores
       Feedback: hit/miss → bandit update → dataset log
       Write Pipeline: distill + chunk + store new memories
```

## The Core Innovation

**Kinematic trajectory in embedding space:**
Every conversation has a position (C_t), velocity (M_t), and acceleration (A_t)
in high-dimensional semantic space. PPM predicts where the conversation will be
in k steps and prefetches memories from that predicted location — before the
user asks.

**Self-improving via Bayesian bandits:**
Every prediction outcome updates a Beta-Bernoulli bandit.
The system knows its own confidence and gets smarter every turn.

## Quickstart
