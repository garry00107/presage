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

