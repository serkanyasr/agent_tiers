import logging
from typing import Dict, Any, AsyncGenerator, Callable
from pydantic_ai import Agent
from ...domain.ports.services import AgentService
from ...domain.agent.model import  AgentContext, AgentResponse
import yaml
import os
from ..llm_providers.providers import get_openai_model
from ..mcp.mcp_client import load_mcp_servers
from ..config import settings

logger = logging.getLogger(__name__)


class PydanticAIAgentService(AgentService):
    """Pydantic AI-based AgentService implementation"""
    
    def __init__(self):
        self._servers = None
        self._agent_factory = None
        self._system_prompt = self._load_system_prompt()
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from YAML file."""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            prompt_file = os.path.join(base_dir, "domain", "prompts", "base.yaml")
            
            with open(prompt_file, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
                return data.get('system_prompt', '')
        except Exception as e:
            logger.warning(f"Failed to load system prompt: {e}")
            return "You are a helpful AI assistant."
    
    async def _load_mcp_servers(self) -> bool:
        """Load MCP servers from configuration file."""
        if self._servers is None:
            try:
                config_path = settings.BASE_DIR / "infrastructure" / "mcp" / "mcp_config.json"
                self._servers = await load_mcp_servers(config_path)
                logger.info(f"Loaded {len(self._servers) if self._servers else 0} MCP servers")
                return self._servers is not None
            except Exception as e:
                logger.error(f"Failed to load MCP servers: {e}")
                return False
        return True
    
    def _create_agent_factory(self) -> Callable[[], Agent]:
        """Create agent factory function."""
        if self._agent_factory is None:
            async def _factory() -> Agent:
                await self._load_mcp_servers()
                
                
                agent = Agent(
                    get_openai_model(),
                    toolsets=list(self._servers.values()) if self._servers else [],
                    system_prompt=self._system_prompt
                )
                
                logger.info("Created Pydantic AI agent with MCP toolsets")
                return agent
            
            self._agent_factory = _factory
        
        return self._agent_factory
    
    async def create_agent(self, config: Dict[str, Any]) -> Agent:
        """Create a new agent instance."""
        factory = self._create_agent_factory()
        return await factory()
    
    async def execute_agent(self, agent: Agent, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent with given message and context."""
        try:
            # Convert context to AgentContext
            agent_context = AgentContext(
                session_id=context.get("session_id", ""),
                user_id=context.get("user_id"),
                conversation_history=context.get("conversation_history", []),
                retrieved_documents=context.get("retrieved_documents", []),
                metadata=context.get("metadata", {})
            )
            
            # Prepare prompt
            full_prompt = self._prepare_prompt(message, agent_context)
            
            # Execute agent
            result = await agent.run(full_prompt, deps=context)
            
            # Prepare response - check PydanticAI result object
            logger.debug(f"Result object type: {type(result)}")
            logger.debug(f"Result object attributes: {dir(result)}")
            
            # Get content from PydanticAI result object
            content = result.output
            
            response = AgentResponse(
                content=content,
                session_id=agent_context.session_id,
                tools_used=self._extract_tool_calls(result),
                retrieved_sources=agent_context.retrieved_documents,
                metadata=agent_context.metadata
            )
            
            return response.__dict__
            
        except Exception as e:
            logger.error(f"Error executing agent: {e}")
            raise
    
    async def stream_agent(self, agent: Agent, message: str, context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute agent in streaming mode."""
        try:
            # Convert context to AgentContext
            agent_context = AgentContext(
                session_id=context.get("session_id", ""),
                user_id=context.get("user_id"),
                conversation_history=context.get("conversation_history", []),
                retrieved_documents=context.get("retrieved_documents", []),
                metadata=context.get("metadata", {})
            )
            
            # Prepare prompt
            full_prompt = self._prepare_prompt(message, agent_context)            
            # Streaming response
            full_response = ""
            async with agent.iter(full_prompt, deps=context) as run:
                async for node in run:
                    if agent.is_model_request_node(node):
                        async with node.stream(run.ctx) as request_stream:
                            async for event in request_stream:
                                if hasattr(event, 'part') and hasattr(event.part, 'content'):
                                    delta_content = event.part.content
                                    yield {"type": "text", "content": delta_content}
                                    full_response += delta_content
                                elif hasattr(event, 'delta') and hasattr(event.delta, 'content_delta'):
                                    delta_content = event.delta.content_delta
                                    yield {"type": "text", "content": delta_content}
                                    full_response += delta_content
            
            # Extract tool calls
            result = run.result
            tools_used = self._extract_tool_calls(result)
            
            if tools_used:
                yield {"type": "tools", "tools": tools_used}
            
            yield {"type": "end"}
            
        except Exception as e:
            logger.error(f"Error streaming agent: {e}")
            yield {"type": "error", "error": str(e)}
    
    def get_agent_factory(self) -> Callable[[], Agent]:
        """Get agent factory function."""
        return self._create_agent_factory()
    
    def _prepare_prompt(self, message: str, context: AgentContext) -> str:
        """Prepare prompt with context and retrieved documents."""
        full_prompt = message
        
        # If retrieved documents exist, add to context
        if context.retrieved_documents:
            docs_text = "\n".join([
                f"Document: {doc.get('title', 'Unknown')}\nContent: {doc.get('content', '')}"
                for doc in context.retrieved_documents
            ])
            full_prompt = f"Relevant documents:\n{docs_text}\n\nQuestion: {message}"
        
        # If conversation history exists, add to context
        elif context.conversation_history:
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in context.conversation_history[-6:]  # Last 6 messages
            ])
            full_prompt = f"Previous conversation:\n{history_text}\n\nCurrent question: {message}"
        
        return full_prompt
    
    def _extract_tool_calls(self, result) -> list[Dict[str, Any]]:
        """Extract tool calls from agent result."""
        tools_used = []
        try:
            messages = result.all_messages()
            for message in messages:
                if hasattr(message, "parts"):
                    for part in message.parts:
                        if part.__class__.__name__ == "ToolCallPart":
                            tool_name = str(getattr(part, 'tool_name', 'unknown'))
                            tool_args = {}
                            if hasattr(part, "args") and part.args is not None:
                                if isinstance(part.args, str):
                                    try:
                                        import json
                                        tool_args = json.loads(part.args)
                                    except json.JSONDecodeError:
                                        pass
                                elif isinstance(part.args, dict):
                                    tool_args = part.args
                            tool_call_id = str(part.tool_call_id) if hasattr(part, "tool_call_id") and part.tool_call_id else None
                            tools_used.append({
                                "tool_name": tool_name,
                                "args": tool_args,
                                "tool_call_id": tool_call_id
                            })
        except Exception as e:
            logger.warning(f"Failed to extract tool calls: {e}")
        
        return tools_used
