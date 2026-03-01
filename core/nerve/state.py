"""
ConversationStateManager — maintains the mutable trajectory state
across turns.

Owns:
  - The turn embedding history (bounded deque, max N=6)
  - The current conversation state vector C_t
  - The smoothed momentum M_hat and raw momentum M_raw
  - The velocity and acceleration scalars

The state manager is the only mutable object in the Nerve Layer.
Everything else is stateless computation.

Thread safety: this runs in a single asyncio event loop per session.
No locking needed — asyncio is cooperative, not preemptive.
"""

from collections import deque

import numpy as np

from core.nerve.models import ConversationState, IntentSignal
from core.types import UnitVector
from math_core.momentum import (
    conversation_state,
    l2_normalize,
    momentum_tangent,
    predict_future_state,
)
from config.settings import settings


class ConversationStateManager:
    """
    Manages the evolving trajectory state for one session.
    One instance per active session — created by the session factory.
    """

    def __init__(self, dim: int):
        self._dim = dim
        self._history: deque[UnitVector] = deque(maxlen=settings.state_window_max)
        self._C_t:  UnitVector | None = None       # current state
        self._C_prev: UnitVector | None = None     # previous state
        self._M_hat: UnitVector | None = None      # smoothed tangent direction
        self._M_raw: np.ndarray | None = None      # raw momentum (for acceleration)
        self._velocity: float = 0.0
        self._velocity_prev: float = 0.0           # for acceleration
        self._turn_count: int = 0
        self._last_intent: IntentSignal = IntentSignal.UNKNOWN

    # ── Public API ─────────────────────────────────────────────────────────────

    def push(self, embedding: UnitVector, lambda_effective: float) -> None:
        """
        Ingest a new turn embedding and update all state vectors.
        Called by ConversationObserver after every turn.
        """
        self._history.append(embedding)
        self._turn_count += 1

        # Recompute conversation state C_t from history
        C_new = conversation_state(list(self._history), decay=lambda_effective)

        if self._C_t is not None:
            # Update momentum
            self._C_prev = self._C_t
            if len(self._history) > 1: # Momentum calculations only if history size > 1
                M_hat_new, vel_new = momentum_tangent(
                    C_t=C_new,
                    C_prev=self._C_t,
                    M_prev=self._M_hat,
                    beta=settings.momentum_beta,
                )
                # Acceleration = change in velocity
                self._velocity_prev = self._velocity
                self._velocity = vel_new
                self._M_hat = M_hat_new
            else: # Zeroize momentum if history size is 1 (e.g., after a reset)
                self._velocity = 0.0
                self._velocity_prev = 0.0
                self._M_hat = None
                self._M_raw = None # Also zeroize raw momentum
        else:
            # First turn: no momentum yet
            self._velocity = 0.0

        self._C_t = C_new

    def predict(self, k: int = 1) -> UnitVector:
        """
        Geodesic extrapolation: where will the conversation be in k turns?
        Returns the predicted state as a UnitVector (safe query vector).
        """
        if self._C_t is None:
            return l2_normalize(np.zeros(self._dim))

        if self._M_hat is None or self._velocity < 1e-8:
            # No momentum yet — predict current state (no movement)
            return self._C_t

        return predict_future_state(
            C_t=self._C_t,
            M_hat=self._M_hat,
            velocity=self._velocity,
            k=k,
            step_size=settings.slerp_step_size,
        )

    def reset(self) -> None:
        """
        Hard reset on context switch. Clears history and momentum.
        Called by ConversationObserver when switch_score > threshold.
        """
        self._history.clear()
        self._C_prev = None
        self._M_hat = None
        self._M_raw = None
        self._velocity = 0.0
        self._velocity_prev = 0.0
        # Note: _C_t and _turn_count intentionally NOT reset —
        # we keep the last known position as the new starting point.

    def snapshot(self) -> ConversationState:
        """Serializable snapshot of current state for persistence/logging."""
        return ConversationState(
            C_t=self._C_t if self._C_t is not None
                else l2_normalize(np.zeros(self._dim)),
            M_hat=self._M_hat,
            M_raw=self._M_raw,
            velocity=self._velocity,
            acceleration=self._velocity - self._velocity_prev,
            turn_count=self._turn_count,
            lambda_effective=settings.decay_lambda_base,
            last_intent=self._last_intent,
            did_reset_last=False,
        )

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def current_velocity(self) -> float:
        return self._velocity

    @property
    def acceleration(self) -> float:
        return self._velocity - self._velocity_prev

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def has_momentum(self) -> bool:
        return self._M_hat is not None and self._velocity > 1e-8

    @property
    def C_t(self) -> UnitVector | None:
        return self._C_t

