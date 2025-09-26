import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from ...domain.ports.repositories import SessionRepository, MessageRepository
from ...domain.ports.services import RetrievalService, AgentService

logger = logging.getLogger(__name__)


@dataclass
class ChatDependencies:
    sessions: SessionRepository
    messages: MessageRepository
    agent_service: AgentService
    retrieval: Optional[RetrievalService] = None


class ChatUseCase:
    """Use case for non-streaming chat interactions."""
    
    def __init__(self, deps: ChatDependencies):
        """Initialize chat use case with dependencies."""
        self.deps = deps

    def _extract_tool_calls(self, result) -> list[dict]:
        """Extract tool calls from agent result."""
        tools_used: list[dict] = []
        try:
            messages = result.all_messages()
            for message in messages:
                if hasattr(message, "parts"):
                    for part in message.parts:
                        if part.__class__.__name__ == "ToolCallPart":
                            tool_name = str(getattr(part, "tool_name", "unknown"))
                            tool_args = {}
                            if hasattr(part, "args") and part.args is not None:
                                if isinstance(part.args, str):
                                    try:
                                        import json as _json
                                        tool_args = _json.loads(part.args)
                                    except Exception:
                                        tool_args = {}
                                elif isinstance(part.args, dict):
                                    tool_args = part.args
                            tool_call_id = str(getattr(part, "tool_call_id", "")) or None
                            tools_used.append({
                                "tool_name": tool_name,
                                "args": tool_args,
                                "tool_call_id": tool_call_id,
                            })
        except Exception:
            pass
        return tools_used

    async def get_or_create_session(self, session_id: Optional[str], user_id: Optional[str], metadata: Dict[str, Any]) -> str:
        """Get existing session or create new one."""
        if session_id:
            session = await self.deps.sessions.get(session_id)
            if session:
                return session_id
        return await self.deps.sessions.create(user_id=user_id, metadata=metadata)

    async def get_context(self, session_id: str, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get conversation context for given session."""
        messages = await self.deps.messages.list(session_id, limit=max_messages)
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    async def run(self, message: str, session_id: Optional[str], user_id: Optional[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Execute chat use case with message and context."""
        current_session_id = await self.get_or_create_session(session_id, user_id, metadata)
        
        context_messages = await self.get_context(current_session_id)
        
        retrieved_sources = []
        if self.deps.retrieval:
            try:
                retrieved_sources = await self.deps.retrieval.hybrid_search(message, limit=3, text_weight=0.3)
                logger.info(f"Retrieved {len(retrieved_sources)} sources for prompt")
            except Exception as e:
                logger.warning(f"Retrieval failed: {e}")
        
        agent_context = {
            "session_id": current_session_id,
            "user_id": user_id,
            "conversation_history": context_messages,
            "retrieved_documents": retrieved_sources,
            "metadata": metadata
        }
        
        agent = await self.deps.agent_service.create_agent({})
        result = await self.deps.agent_service.execute_agent(agent, message, agent_context)
        
        await self.deps.messages.add(current_session_id, "user", message, metadata)
        await self.deps.messages.add(current_session_id, "assistant", result["content"], metadata)
        
        return {
            "message": result["content"],
            "session_id": current_session_id,
            "tools_used": result.get("tools_used", []),
            "retrieved_sources": result.get("retrieved_sources", [])
        }


