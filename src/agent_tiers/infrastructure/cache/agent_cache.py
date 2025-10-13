from typing import Dict, Any, Optional
from pydantic_ai import Agent
import logging

logger = logging.getLogger(__name__)

class AgentCache:
    """Singleton cache for agent instances"""
    _instance = None
    _agents: Dict[str, Agent] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentCache, cls).__new__(cls)
        return cls._instance

    def get(self, session_id: str) -> Optional[Agent]:
        """Get agent for session"""
        return self._agents.get(session_id)

    def set(self, session_id: str, agent: Agent):
        """Set agent for session"""
        self._agents[session_id] = agent
        logger.info(f"Cached agent for session {session_id}")

    def remove(self, session_id: str):
        """Remove agent for session"""
        if session_id in self._agents:
            del self._agents[session_id]
            logger.info(f"Removed agent for session {session_id}")

    def clear(self):
        """Clear all cached agents"""
        self._agents.clear()
        logger.info("Cleared all cached agents")