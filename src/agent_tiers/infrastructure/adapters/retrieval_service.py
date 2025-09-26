from typing import List, Dict, Any

from src.agent_tiers.domain.ports.services import RetrievalService
from src.agent_tiers.infrastructure.db.rag_db import (
    db_pool,
    vector_search as db_vector_search,
    hybrid_search as db_hybrid_search,
)
from src.agent_tiers.infrastructure.llm_providers.providers import (
    get_openai_embedding_client,
    get_openai_embedding_model,
)


class DBRetrievalService(RetrievalService):
    def __init__(self) -> None:
        self._embedding_client = get_openai_embedding_client()
        self._embedding_model = get_openai_embedding_model()

    async def _embed(self, text: str) -> List[float]:
        response = await self._embedding_client.embeddings.create(
            model=self._embedding_model,
            input=text,
        )
        return response.data[0].embedding

    async def vector_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        embedding = await self._embed(query)
        return await db_vector_search(pool=db_pool.pool, embedding=embedding, limit=limit)

    async def hybrid_search(self, query: str, limit: int = 10, text_weight: float = 0.3) -> List[Dict[str, Any]]:
        embedding = await self._embed(query)
        return await db_hybrid_search(
            pool=db_pool.pool,
            embedding=embedding,
            query_text=query,
            limit=limit,
            text_weight=text_weight,
        )
