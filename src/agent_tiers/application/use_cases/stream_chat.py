import logging
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Any, List, Optional
from ...domain.ports.repositories import SessionRepository, MessageRepository
from ...domain.ports.services import AgentService

logger = logging.getLogger(__name__)


@dataclass
class StreamChatDeps:
    sessions: SessionRepository
    messages: MessageRepository
    agent_service: AgentService


class StreamChatUseCase:
    """Use case for streaming chat interactions."""
    
    def __init__(self, deps: StreamChatDeps):
        """Initialize stream chat use case with dependencies."""
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

    async def stream(self, message: str, session_id: Optional[str], user_id: Optional[str], metadata: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Execute streaming chat use case with message and context."""
        current_session_id = await self.get_or_create_session(session_id, user_id, metadata)
        yield {"type": "session", "session_id": current_session_id}
        
        context_messages = await self.get_context(current_session_id)
        
        agent_context = {
            "session_id": current_session_id,
            "user_id": user_id,
            "conversation_history": context_messages,
            "retrieved_documents": [],
            "metadata": metadata
        }
        
        await self.deps.messages.add(current_session_id, "user", message, {**metadata, "user_id": user_id})
        
        agent = await self.deps.agent_service.create_agent({})
        full_response = ""
        tools_used = []
        
        async for event in self.deps.agent_service.stream_agent(agent, message, agent_context):
            if event["type"] == "text":
                full_response += event["content"]
                yield event
            elif event["type"] == "tools":
                tools_used = event.get("tools", [])
                yield event
            elif event["type"] == "error":
                logger.error(f"Agent streaming error: {event['error']}")
                yield event
            elif event["type"] == "end":
                break

        await self.deps.messages.add(current_session_id, "assistant", full_response, {"streamed": True, "tool_calls": len(tools_used)})
        yield {"type": "end"}


