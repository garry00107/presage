"""
Local embedding adapter via sentence-transformers.
Supports nomic-embed-text and bge-m3. No API key required.
Install: pip install sentence-transformers
"""

import numpy as np
from core.types import UnitVector
from adapters.embedder.base import Embedder


class LocalEmbedder(Embedder):
    """
    Local embedder using sentence-transformers.
    Lazy-loads model on first use.
    """

    MODEL_DIMS = {
        "nomic-ai/nomic-embed-text-v1.5": 768,
        "BAAI/bge-m3": 1024,
    }

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self._model_name = model_name
        self._model = None  # lazy load
        self._dim = self.MODEL_DIMS.get(model_name, 768)

    @property
    def dim(self) -> int:
        return self._dim

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, trust_remote_code=True)

    async def embed(self, text: str) -> UnitVector:
        self._load()
        import asyncio
        loop = asyncio.get_event_loop()
        v = await loop.run_in_executor(
            None, lambda: self._model.encode(text, normalize_embeddings=True)
        )
        return self._normalize(np.array(v, dtype=np.float32))

    async def embed_batch(self, texts: list[str]) -> list[UnitVector]:
        self._load()
        import asyncio
        loop = asyncio.get_event_loop()
        vecs = await loop.run_in_executor(
            None, lambda: self._model.encode(texts, normalize_embeddings=True,
                                             batch_size=32)
        )
        return [self._normalize(np.array(v, dtype=np.float32)) for v in vecs]

