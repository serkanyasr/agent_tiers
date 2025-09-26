from typing import Protocol, List, Dict, Any, Optional, Callable, AsyncGenerator


class RetrievalService(Protocol):
    async def vector_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]: ...
    async def hybrid_search(self, query: str, limit: int = 10, text_weight: float = 0.3) -> List[Dict[str, Any]]: ...


class EmbeddingService(Protocol):
    async def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]: ...


class ExtractionService(Protocol):
    def extract(self, file_path: str) -> Dict[str, Any]: ...


class MemoryService(Protocol):
    async def save(self, user_id: str, text: str) -> None: ...
    async def search(self, user_id: str, query: str, limit: int = 3) -> List[str]: ...


class IngestionService(Protocol):
    async def ingest_documents(
        self,
        documents_folder: str,
        output_folder: str,
        config: Dict[str, Any],
        clean_before_ingest: bool = False,
    ) -> List[Dict[str, Any]]: ...


class AgentService(Protocol):
    
    async def create_agent(self, config: Dict[str, Any]) -> Any: ...
    async def execute_agent(self, agent: Any, message: str, context: Dict[str, Any]) -> Dict[str, Any]: ...
    async def stream_agent(self, agent: Any, message: str, context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]: ...
    def get_agent_factory(self) -> Callable[[], Any]: ...

