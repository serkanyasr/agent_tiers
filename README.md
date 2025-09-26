# Agent Tiers

A sophisticated AI agent system built with layered architecture, featuring RAG (Retrieval-Augmented Generation), memory management, and MCP (Model Context Protocol) integration.

## 🚀 Features

### Core Capabilities
- **RAG System**: Vector search and semantic matching for document retrieval
- **Memory Management**: Persistent user memory using Mem0
- **MCP Integration**: Extensible tool integration through Model Context Protocol
- **Streaming Chat**: Real-time response streaming with Server-Sent Events
- **Document Processing**: Support for multiple file formats with semantic chunking
- **Session Management**: Conversation context across multiple sessions

### Architecture
- **Domain-Driven Design**: Clean separation of business logic
- **Layered Architecture**: Domain, Application, Infrastructure, and Interface layers
- **Ports & Adapters**: Flexible dependency injection and testability
- **Docker Support**: Complete containerization with Docker Compose

## 🏗️ Architecture

### System Overview
```mermaid
graph TB
    subgraph "Client Layer"
        UI[Streamlit UI<br/>Port: 8501]
        API_CLIENT[API Client<br/>HTTP/SSE]
    end
    
    subgraph "API Layer"
        FASTAPI[FastAPI Server<br/>Port: 8000]
        ENDPOINTS["/chat<br/>/chat/stream<br/>/documents/upload<br/>/health"]
    end
    
    subgraph "Application Layer"
        USE_CASES[Use Cases<br/>ChatUseCase<br/>StreamChatUseCase<br/>UploadDocumentsUseCase]
        ORCHESTRATOR[Agent Orchestrator<br/>Agent Management]
    end
    
    subgraph "Domain Layer"
        PORTS[Ports/Interfaces<br/>AgentService<br/>RetrievalService<br/>MemoryService]
        ENTITIES[Domain Entities<br/>Agent<br/>AgentConfig<br/>AgentContext]
    end
    
    subgraph "Infrastructure Layer"
        ADAPTERS[Adapters<br/>PydanticAIAgentService<br/>DBRetrievalService<br/>AsyncpgRepositories]
        MCP_SERVERS[MCP Servers<br/>Memory MCP: 8050<br/>RAG MCP: 8055]
    end
    
    subgraph "Data Layer"
        MEMORY_DB[(Memory DB<br/>PostgreSQL + pgvector<br/>Port: 5432)]
        RAG_DB[(RAG DB<br/>PostgreSQL + pgvector<br/>Port: 5433)]
    end
    
    subgraph "External Services"
        OPENAI[OpenAI API<br/>GPT + Embeddings]
    end
    
    UI --> FASTAPI
    API_CLIENT --> FASTAPI
    FASTAPI --> ENDPOINTS
    ENDPOINTS --> USE_CASES
    USE_CASES --> ORCHESTRATOR
    ORCHESTRATOR --> PORTS
    PORTS --> ADAPTERS
    ADAPTERS --> MCP_SERVERS
    ADAPTERS --> MEMORY_DB
    ADAPTERS --> RAG_DB
    ADAPTERS --> OPENAI
```

### Directory Structure
```
src/agent_tiers/
├── domain/                 # Business logic and entities
│   ├── agent/             # Agent domain models
│   ├── prompts/           # Prompt templates
│   └── ports/             # Interface definitions
├── application/           # Use cases and orchestration
│   ├── agent_service/     # Agent orchestration
│   └── use_cases/         # Business use cases
├── infrastructure/        # External integrations
│   ├── adapters/          # Port implementations
│   ├── api/              # FastAPI endpoints
│   ├── db/               # Database connections
│   ├── mcp/              # MCP server implementations
│   └── ui/               # Streamlit interface
└── tests/                # Test suites
```

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.11+
- **Database**: PostgreSQL with pgvector
- **AI/ML**: Pydantic AI, OpenAI GPT
- **Memory**: Mem0 for persistent memory
- **RAG**: Vector embeddings with semantic search
- **MCP**: Model Context Protocol for tool integration
- **UI**: Streamlit
- **Containerization**: Docker & Docker Compose

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- OpenAI API key

### 1. Clone and Setup
```bash
git clone <repository-url>
cd agent_tiers
```

### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Start Services
```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps
```

### 4. Access the Application
- **API**: http://localhost:8000
- **UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

## 📖 API Endpoints

### Chat Endpoints
- `POST /chat` - Non-streaming chat
- `POST /chat/stream` - Streaming chat with SSE

### Document Management
- `POST /documents/upload` - Upload and process documents

### Health & Status
- `GET /health` - Service health check

## 🔄 System Flow Diagrams

### Streaming Chat Flow
```mermaid
sequenceDiagram
    participant U as User
    participant UI as Streamlit UI
    participant API as FastAPI
    participant SC as StreamChatUseCase
    participant AS as AgentService
    participant MCP as MCP Servers
    
    U->>UI: Send message
    UI->>API: POST /chat/stream
    API->>SC: Execute streaming use case
    
    SC->>AS: Create agent
    AS->>MCP: Load MCP servers
    
    loop Streaming Response
        SC->>AS: Stream agent
        AS->>OPENAI: Stream request
        OPENAI-->>AS: Stream delta
        AS-->>SC: Text delta
        SC-->>API: SSE event
        API-->>UI: Server-Sent Event
        UI-->>U: Display partial text
    end
    
    SC->>AS: Extract tool calls
    AS-->>SC: Tools used
    SC-->>API: Final response
    API-->>UI: End event
    UI-->>U: Complete message
```

### Document Upload Flow
```mermaid
sequenceDiagram
    participant U as User
    participant UI as Streamlit UI
    participant API as FastAPI
    participant UC as UploadUseCase
    participant IS as IngestionService
    participant FS as File System
    participant DB as RAG Database
    
    U->>UI: Upload document
    UI->>API: POST /documents/upload
    API->>API: Validate file type/size
    API->>FS: Save temp file
    
    API->>UC: Execute upload use case
    UC->>IS: Ingest documents
    IS->>FS: Read document
    IS->>IS: Extract text content
    IS->>IS: Create semantic chunks
    IS->>DB: Store embeddings
    DB-->>IS: Chunks stored
    IS-->>UC: Ingestion results
    
    UC-->>API: Upload results
    API->>FS: Clean temp files
    API-->>UI: Success response
    UI-->>U: Show results
```

## 🔧 Configuration

### Environment Variables
```bash
# Database
MEMORY_DB_NAME=memory_db
MEMORY_DB_USER=memory_user
MEMORY_DB_PASSWORD=memory_pass
RAG_DB_NAME=rag_db
RAG_DB_USER=rag_user
RAG_DB_PASSWORD=rag_pass

# OpenAI
OPENAI_API_KEY=your_api_key

# Upload Settings
UPLOAD_MAX_FILE_MB=25
UPLOAD_ALLOWED_EXTENSIONS=.pdf,.docx,.doc,.txt,.md,.png,.jpg,.jpeg

# API
API_ENV=dev
API_HOST=0.0.0.0
API_PORT=8000
```

### MCP Configuration
Edit `src/agent_tiers/infrastructure/mcp/mcp_config.json` to add new MCP servers:

```json
{
  "mcpServers": {
    "memory": {
      "protocol": "http-stream",
      "url": "http://mcp_memory:8050/mcp"
    },
    "rag": {
      "protocol": "http-stream", 
      "url": "http://mcp_rag:8055/mcp"
    }
  }
}
```

## 🧪 Development

### Local Development
```bash
# Install dependencies
pip install uv
uv sync

# 1. Start databases first
docker-compose up -d memory_postgres rag_postgres

# 2. Start MCP servers
docker-compose up -d mcp_memory mcp_rag

# 3. Run API
python -m src.agent_tiers.infrastructure.api.main

# 4. Run UI (in another terminal)
streamlit run src/agent_tiers/infrastructure/ui/app.py
```

## 🐳 Docker Commands

### Development
```bash
# Build and start
docker-compose up --build

# View logs
docker-compose logs -f api

# Restart service
docker-compose restart api
```

### Production
```bash
# Production deployment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Cleanup
```bash
# Stop and remove containers
docker-compose down

# Remove volumes (WARNING: Data loss)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

## 📚 Usage Examples

### Basic Chat
```python
import requests

response = requests.post("http://localhost:8000/chat", json={
    "message": "What is machine learning?",
    "user_id": "user123"
})

print(response.json()["message"])
```

### Streaming Chat
```python
import requests

response = requests.post("http://localhost:8000/chat/stream", 
    json={"message": "Explain quantum computing"},
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode())
```

### Document Upload
```python
import requests

files = {"files": open("document.pdf", "rb")}
response = requests.post("http://localhost:8000/documents/upload", files=files)

print(response.json())
```

## 🔍 System Components

### Domain Layer Architecture
```mermaid
classDiagram
    class Agent {
        +String agent_id
        +AgentConfig config
        +AgentStatus status
        +String created_at
        +String last_used_at
        +int usage_count
        +update_status(status)
        +increment_usage()
        +has_capability(capability)
        +is_available()
    }
    
    class AgentConfig {
        +String model_name
        +float temperature
        +int max_tokens
        +String system_prompt
        +List~AgentCapability~ capabilities
    }
    
    class AgentContext {
        +String session_id
        +String user_id
        +List~Dict~ conversation_history
        +List~Dict~ retrieved_documents
        +Dict metadata
    }
    
    class AgentResponse {
        +String content
        +String session_id
        +List~Dict~ tools_used
        +List~Dict~ retrieved_sources
        +Dict metadata
    }
    
    class AgentCapability {
        <<enumeration>>
        CHAT
        DOCUMENT_ANALYSIS
        MEMORY_MANAGEMENT
        RAG_RETRIEVAL
        TOOL_CALLING
    }
    
    class AgentStatus {
        <<enumeration>>
        IDLE
        PROCESSING
        ERROR
        UNAVAILABLE
    }
    
    Agent --> AgentConfig
    Agent --> AgentStatus
    Agent --> AgentCapability
    AgentContext --> Agent
    AgentResponse --> Agent
```

### Ports and Adapters Pattern
```mermaid
graph LR
    subgraph "Domain Layer"
        P1[AgentService Port]
        P2[RetrievalService Port]
        P3[MemoryService Port]
        P4[SessionRepository Port]
        P5[MessageRepository Port]
    end
    
    subgraph "Infrastructure Layer"
        A1[PydanticAIAgentService]
        A2[DBRetrievalService]
        A3[Mem0MemoryService]
        A4[AsyncpgSessionRepository]
        A5[AsyncpgMessageRepository]
    end
    
    subgraph "External Systems"
        E1[OpenAI API]
        E2[PostgreSQL + pgvector]
        E3[MCP Servers]
    end
    
    P1 -.-> A1
    P2 -.-> A2
    P3 -.-> A3
    P4 -.-> A4
    P5 -.-> A5
    
    A1 --> E1
    A1 --> E3
    A2 --> E2
    A3 --> E2
    A4 --> E2
    A5 --> E2
```

### MCP Server Communication
```mermaid
sequenceDiagram
    participant A as Agent
    participant MCP as MCP Client
    participant MEM as Memory MCP
    participant RAG as RAG MCP
    participant DB as Database
    
    A->>MCP: Tool call request
    MCP->>MCP: Route to appropriate server
    
    alt Memory Operation
        MCP->>MEM: Memory tool call
        MEM->>DB: Query/Update memory
        DB-->>MEM: Memory data
        MEM-->>MCP: Memory result
    else RAG Operation
        MCP->>RAG: RAG tool call
        RAG->>DB: Vector search
        DB-->>RAG: Retrieved documents
        RAG-->>MCP: RAG result
    end
    
    MCP-->>A: Tool response
```

### Data Flow Architecture
```mermaid
flowchart TD
    subgraph "Input Sources"
        USER[User Input]
        DOCS[Document Uploads]
    end
    
    subgraph "Processing Layer"
        CHAT[Chat Processing]
        UPLOAD[Document Ingestion]
        RAG[Vector Search]
        MEM[Memory Management]
    end
    
    subgraph "Storage Layer"
        VECTOR[(Vector DB<br/>Embeddings)]
        SESSION[(Session DB<br/>Conversations)]
        MEMORY[(Memory DB<br/>User Data)]
    end
    
    subgraph "AI Layer"
        LLM[Large Language Model]
        EMBED[Embedding Model]
    end
    
    USER --> CHAT
    DOCS --> UPLOAD
    CHAT --> RAG
    CHAT --> MEM
    UPLOAD --> VECTOR
    RAG --> VECTOR
    MEM --> MEMORY
    CHAT --> SESSION
    
    RAG --> EMBED
    CHAT --> LLM
    MEM --> LLM
    
    VECTOR --> RAG
    SESSION --> CHAT
    MEMORY --> MEM
```

### RAG System
- **Vector Database**: PostgreSQL with pgvector extension
- **Embeddings**: OpenAI text-embedding-ada-002
- **Search**: Hybrid search (semantic + keyword)
- **Chunking**: Semantic chunking for optimal retrieval

### Memory System
- **Storage**: Mem0 for persistent user memory
- **Context**: Session-based conversation context
- **Personalization**: User preference learning
- **Privacy**: Secure data handling

### MCP Integration
- **Memory Server**: User data storage and retrieval
- **RAG Server**: Document search and analysis
- **Extensible**: Easy addition of new MCP servers
- **Protocol**: HTTP-stream for real-time communication

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🗺️ Roadmap

- [ ] **Graph Database Integration**: Neo4j or similar for complex relationship modeling
- [ ] **Additional MCP Server Integrations**: Web search, calendar, email, and more
- [ ] **Real-time Collaboration**: Multi-user chat sessions and shared workspaces
- [ ] **Multi-modal AI Integration**: Image generation, voice synthesis, and video processing
