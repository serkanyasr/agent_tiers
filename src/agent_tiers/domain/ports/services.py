from typing import Protocol, List, Dict, Any, Callable, AsyncGenerator


class RetrievalService(Protocol):
    async def vector_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]: ...
    async def hybrid_search(self, query: str, limit: int = 10, text_weight: float = 0.3) -> List[Dict[str, Any]]: ...


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

