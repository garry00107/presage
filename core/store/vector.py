"""
Qdrant Vector Store Adapter for PPM.

Stores chunks (not raw memories) as vectors. 
Uses UnitVectors derived from math_core.momentum.l2_normalize.

Supports async operations via qdrant_client.AsyncQdrantClient.
"""

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

import numpy as np
from core.types import ChunkID, MemoryID, Chunk

class QdrantVectorStore:
    """Async adapter for Qdrant."""
    
    def __init__(self, client: AsyncQdrantClient, collection_name: str = "ppm_chunks", dim: int = 1536):
        self.client = client
        self.collection_name = collection_name
        self.dim = dim
        
    async def initialize(self):
        """Creates the collection if it doesn't exist."""
        collections = await self.client.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)
        
        if not exists:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dim,
                    distance=models.Distance.COSINE # All vectors are UnitVectors
                )
            )

    async def upsert(self, chunk_id: str, parent_id: str, content: str, tokens: int, source_type: str, vector: list[float]):
        """Upsert a chunk."""
        payload = {
            "parent_id": parent_id,
            "content": content,
            "tokens": tokens,
            "source_type": source_type
        }
        
        await self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )

    async def delete(self, memory_id: str):
        """Delete all chunks belonging to a single memory."""
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="parent_id",
                        match=models.MatchValue(value=memory_id)
                    )
                ]
            )
        )

    async def search(self, query_vector: np.ndarray, top_k: int = 20, filter_dict: dict = None) -> list[Chunk]:
        """
        Cosine similarity search.
        query_vector must be a UnitVector.
        """
        qdrant_filter = None
        if filter_dict:
            must_conditions = []
            for k, v in filter_dict.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=k,
                        match=models.MatchValue(value=v)
                    )
                )
            qdrant_filter = models.Filter(must=must_conditions)

        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=qdrant_filter
        )
        
        chunks = []
        for scored_point in results.points:
            chunks.append({
                "id": ChunkID(str(scored_point.id)),
                "parent_id": MemoryID(scored_point.payload.get("parent_id")),
                "chunk_index": scored_point.payload.get("chunk_index", 0), # Optional in payload structure
                "content": scored_point.payload.get("content", ""),
                "tokens": scored_point.payload.get("tokens", 0),
                "source_type": scored_point.payload.get("source_type", "prose"),
                "score": scored_point.score,
                "embedding": None # don't return embedding on search unless needed
            })
            
        return chunks
