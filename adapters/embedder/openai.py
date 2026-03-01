import numpy as np
from openai import AsyncOpenAI
from core.types import UnitVector
from adapters.embedder.base import Embedder
from config.settings import settings


class OpenAIEmbedder(Embedder):
    """
    OpenAI text-embedding-3-small (or -large) adapter.
    Returns L2-normalized UnitVectors.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None):
        self._client = AsyncOpenAI(
            api_key=api_key or settings.openai_api_key,
            base_url=base_url
        )
        self._model = model or settings.embedder_model
        self._dim = settings.embedder_dim
        self._is_nvidia = base_url is not None and "nvidia" in base_url

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, text: str) -> UnitVector:
        kwargs = {}
        if self._is_nvidia:
            kwargs["extra_body"] = {"input_type": "query", "truncate": "END"}
            text = text[:1800]  # Hard fallback for 512 token limit

        resp = await self._client.embeddings.create(
            model=self._model,
            input=text,
            encoding_format="float",
            **kwargs
        )
        v = np.array(resp.data[0].embedding, dtype=np.float32)
        return self._normalize(v)

    async def embed_batch(self, texts: list[str]) -> list[UnitVector]:
        if not texts:
            return []
            
        kwargs = {}
        if self._is_nvidia:
            kwargs["extra_body"] = {"input_type": "passage", "truncate": "END"}
            results = []
            for t in texts:
                safe_text = t[:1500]
                resp = await self._client.embeddings.create(
                    model=self._model,
                    input=safe_text,
                    encoding_format="float",
                    **kwargs
                )
                v = np.array(resp.data[0].embedding, dtype=np.float32)
                results.append(self._normalize(v))
            return results

        resp = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            encoding_format="float",
        )
        # API returns results in order
        return [
            self._normalize(np.array(item.embedding, dtype=np.float32))
            for item in resp.data
        ]

