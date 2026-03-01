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
