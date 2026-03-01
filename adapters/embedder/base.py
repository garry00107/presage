from abc import ABC, abstractmethod
import numpy as np
from core.types import UnitVector
from math_core.momentum import l2_normalize


class Embedder(ABC):
    """
    Protocol for all embedding backends.
    All implementations MUST return L2-normalized UnitVectors.
    """

    @abstractmethod
    async def embed(self, text: str) -> UnitVector:
        """Embed a single string. Returns UnitVector."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[UnitVector]:
        """Embed multiple strings. More efficient than N single calls."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""
        ...

    def _normalize(self, v: np.ndarray) -> UnitVector:
        """Shared normalization. Call this in all subclass implementations."""
        return l2_normalize(v)

