from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum


class AgentCapability(Enum):
    """Defines agent capabilities."""
    CHAT = "chat"
    DOCUMENT_ANALYSIS = "document_analysis"
    MEMORY_MANAGEMENT = "memory_management"
    RAG_RETRIEVAL = "rag_retrieval"
    TOOL_CALLING = "tool_calling"


class AgentStatus(Enum):
    """Defines agent statuses."""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


@dataclass
class AgentConfig:
    """Agent configuration."""
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    capabilities: List[AgentCapability] = None
    tools_enabled: bool = True
    memory_enabled: bool = True
    rag_enabled: bool = True
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = [AgentCapability.CHAT]


@dataclass
class AgentContext:
    """Agent working context."""
    session_id: str
    user_id: Optional[str] = None
    conversation_history: List[Dict[str, str]] = None
    retrieved_documents: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.retrieved_documents is None:
            self.retrieved_documents = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgentResponse:
    """Agent response."""
    content: str
    session_id: str
    tools_used: List[Dict[str, Any]] = None
    retrieved_sources: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    processing_time_ms: Optional[int] = None
    
    def __post_init__(self):
        if self.tools_used is None:
            self.tools_used = []
        if self.retrieved_sources is None:
            self.retrieved_sources = []
        if self.metadata is None:
            self.metadata = {}


class Agent:
    """Domain Agent entity - contains only data structure, no business logic."""
    
    def __init__(self, agent_id: str, config: AgentConfig):
        """Initialize agent with ID and configuration."""
        self.agent_id = agent_id
        self.config = config
        self.status = AgentStatus.IDLE
        self.created_at: Optional[str] = None
        self.last_used_at: Optional[str] = None
        self.usage_count: int = 0
    
    def update_status(self, status: AgentStatus):
        """Update agent status."""
        self.status = status
    
    def increment_usage(self):
        """Increment usage counter."""
        self.usage_count += 1
    
    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has specific capability."""
        return capability in self.config.capabilities
    
    def is_available(self) -> bool:
        """Check if agent is available for use."""
        return self.status in [AgentStatus.IDLE, AgentStatus.PROCESSING]
