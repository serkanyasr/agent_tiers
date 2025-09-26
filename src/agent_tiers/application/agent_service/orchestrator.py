import logging
from typing import Dict, Any, Callable
from pydantic_ai import Agent
from ...domain.ports.services import AgentService
from ...infrastructure.adapters.agent_service import PydanticAIAgentService

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Domain-driven Agent orchestrator"""
    
    def __init__(self, agent_service: AgentService):
        self.agent_service = agent_service
        self._agent_factory: Callable[[], Agent] = None
    
    async def initialize(self) -> bool:
        """Initialize the orchestrator."""
        try:
            self._agent_factory = self.agent_service.get_agent_factory()
            test_agent = await self._agent_factory()
            logger.info("Agent orchestrator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent orchestrator: {e}")
            return False
    
    async def get_agent_factory(self) -> Callable[[], Agent]:
        """Get agent factory function."""
        if self._agent_factory is None:
            await self.initialize()
        return self._agent_factory
    
    async def create_agent(self, config: Dict[str, Any] = None) -> Agent:
        """Create a new agent instance."""
        if config is None:
            config = {}
        
        factory = await self.get_agent_factory()
        return await factory()
    
    async def execute_agent(self, agent: Agent, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent with given message and context."""
        return await self.agent_service.execute_agent(agent, message, context)
    
    async def stream_agent(self, agent: Agent, message: str, context: Dict[str, Any]):
        """Execute agent in streaming mode."""
        async for event in self.agent_service.stream_agent(agent, message, context):
            yield event
    
    def get_agent_service(self) -> AgentService:
        """Get agent service instance."""
        return self.agent_service
    
    async def load_servers(self) -> bool:
        """Load MCP servers."""
        try:
            if hasattr(self.agent_service, '_load_mcp_servers'):
                return await self.agent_service._load_mcp_servers()
            else:
                logger.warning("Agent service does not support MCP server loading")
                return False
        except Exception as e:
            logger.error(f"Failed to load MCP servers: {e}")
            return False
    
    @property
    def servers(self):
        """Get loaded servers."""
        if hasattr(self.agent_service, '_servers'):
            return self.agent_service._servers
        return None


# Factory function
def create_agent_orchestrator() -> AgentOrchestrator:
    """Create agent orchestrator instance."""
    agent_service = PydanticAIAgentService()
    return AgentOrchestrator(agent_service)
