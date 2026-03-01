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

