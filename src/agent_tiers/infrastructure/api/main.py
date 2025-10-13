import json
import logging
from contextlib import asynccontextmanager
import os
import shutil
import tempfile
from typing import List, Optional
from datetime import datetime
import uuid
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi import FastAPI, UploadFile, File
import uvicorn


from ..ingestion.extract import PDFExtractionConfig
from ..ingestion.ingest import IngestionConfig
from ..adapters.agent_service import PydanticAIAgentService
from ..adapters.repositories_asyncpg import AsyncpgSessionRepository, AsyncpgMessageRepository
from ..adapters.retrieval_service import DBRetrievalService
from ...application.use_cases.chat import ChatUseCase, ChatDependencies
from ...application.use_cases.stream_chat import StreamChatUseCase, StreamChatDeps
from ...application.agent_service.orchestrator import create_agent_orchestrator
from ..config import settings

from ..db.rag_db import (
    execute_init_sql,
    initialize_database,
    close_database,
    test_connection as rag_db_conn_test,
)
from ..db.memory_db import test_db_connection as memory_db_conn_test

from .schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthStatus,
    MessageHistoryResponse,
    MessageInfo,
    SessionInfo,
    SessionListResponse,
    ToolCall
)

from ...application.use_cases.chat import ChatUseCase, ChatDependencies
from ...application.use_cases.stream_chat import StreamChatUseCase, StreamChatDeps
from ...application.use_cases.upload_documents import UploadDocumentsUseCase, UploadDeps
from ...infrastructure.adapters.ingestion_service import PipelineIngestionService

orchestrator = create_agent_orchestrator()

# Initialize global services
agent_service = orchestrator.get_agent_service()

# API configuration
API_ENV = settings.API_ENV
API_HOST = settings.API_HOST
API_PORT = settings.API_PORT
API_LOG_LEVEL = settings.API_LOG_LEVEL


# Configure logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=getattr(logging, API_LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

if API_ENV == "development":
    logger.setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager"""
    # Startup
    logger.info("Starting up system...")

    try:
        # Initialize RAG database connections
        await initialize_database()
        await execute_init_sql(settings.SCHEMA_PATH)

        # Init MCP Servers
        servers_ok = await orchestrator.load_servers()

        if servers_ok:
            logger.info("MCP Server loaded")
            logger.info(orchestrator.servers)
        else:
            logger.info("MCP server was not loaded")


        # Test connections
        rag_db_ok = await rag_db_conn_test()
        memory_db_ok = memory_db_conn_test()

        if not rag_db_ok:
            logger.error("RAG Database connection failed")
        else:
            logger.info("RAG Database initialized")
            
        if not memory_db_ok:
            logger.error("MEMORY Database connection failed")
        else:
            logger.info("MEMORY Database initialized")


    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down system...")

    try:
        await close_database()
        logger.info("Connections closed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# Create FastAPI app
app = FastAPI(
    title="Agent tiers",
    description="AI agent tiers",
    version="1.1.0",
    lifespan=lifespan,
)

# Add middleware with flexible CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)



def cleanup_folder(folder_path: str):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        
        
def progress_callback(current: int, total: int):
    print(f"Progress: {current}/{total} documents processed")


# API Endpoints
@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connections
        rag_db_test = await rag_db_conn_test()
        memory_db_test = memory_db_conn_test()
        # Determine overall status
        if rag_db_test:
            rag_db_status = "healthy"
        else:
            rag_db_status = "unhealthy"

        if memory_db_test:
            memory_db_status = "healthy"
        else:
            memory_db_status = "unhealthy"

        return HealthStatus(
            rag_db_status=rag_db_status,
            memory_db_status=memory_db_status,
            llm_connection=True,
            version="1.1.0",
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint."""
    try:
        request_id = str(uuid.uuid4())
        logger.info(f"/chat request_id={request_id} user_id={request.user_id}")
        # Use case dependencies
        sessions_repo = AsyncpgSessionRepository()
        messages_repo = AsyncpgMessageRepository()
        retrieval_service = DBRetrievalService()

        use_case = ChatUseCase(ChatDependencies(
            sessions=sessions_repo,
            messages=messages_repo,
            agent_service=agent_service,
            retrieval=retrieval_service,
        ))

        result = await use_case.run(
            message=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
            metadata=request.metadata or {},
        )

        return ChatResponse(
            message=result["message"],
            session_id=result["session_id"],
            tools_used=[ToolCall(**t) for t in result.get("tools_used", [])],
            metadata={"search_type": str(request.search_type), "request_id": request_id},
        )

    except Exception as e:
        logger.error(f"Chat endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events."""
    try:
        request_id = str(uuid.uuid4())
        logger.info(f"/chat/stream request_id={request_id} user_id={request.user_id}")
        sessions_repo = AsyncpgSessionRepository()
        messages_repo = AsyncpgMessageRepository()

        use_case = StreamChatUseCase(StreamChatDeps(
            sessions=sessions_repo,
            messages=messages_repo,
            agent_service=agent_service,
        ))

        async def generate_stream():
            try:
                yield f"data: {json.dumps({'type': 'request', 'request_id': request_id})}\n\n"
                async for item in use_case.stream(
                    message=request.message,
                    session_id=request.session_id,
                    user_id=request.user_id,
                    metadata=request.metadata or {},
                ):
                    yield f"data: {json.dumps(item)}\n\n"
            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            },
        )

    except Exception as e:
        logger.error(f"Streaming chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload")
async def upload_document(
    files: List[UploadFile] = File(...),
    x_ingestion_config: Optional[str] = Header(None, alias="X-Ingestion-Config")
):
    saved_files = []
    # Upload security settings (from config)
    ALLOWED_EXTENSIONS = set([ext.strip().lower() for ext in settings.UPLOAD_ALLOWED_EXTENSIONS.split(",") if ext.strip()])
    MAX_FILE_SIZE_MB = settings.UPLOAD_MAX_FILE_MB
    MAX_FILE_SIZE_BYTES = settings.UPLOAD_MAX_FILE_MB * 1024 * 1024
    

    config = None
    if x_ingestion_config:
        try:
            config_data = json.loads(x_ingestion_config)
            config = IngestionConfig(**config_data)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid X-Ingestion-Config header: {str(e)}"
            )
    else:
        config = IngestionConfig(
            use_semantic_chunking=True
        )
    
    # Unique folder name
    unique_id = str(uuid.uuid4())
    upload_dir = os.path.join(tempfile.gettempdir(), f"uploaded_{unique_id}")
    os.makedirs(upload_dir, exist_ok=True)

    try:
        for file in files:
            # Extension validation
            _, ext = os.path.splitext(file.filename)
            ext = ext.lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"File type not allowed: {ext}")

            # Normalize file name (only filename, prevent path traversal)
            safe_name = os.path.basename(file.filename)
            file_path = os.path.join(upload_dir, safe_name)

            # Size validation and safe writing
            total_written = 0
            try:
                with open(file_path, "wb") as buffer:
                    while True:
                        chunk = await file.read(1024 * 1024)  # 1MB chunk
                        if not chunk:
                            break
                        total_written += len(chunk)
                        if total_written > MAX_FILE_SIZE_BYTES:
                            buffer.close()
                            os.remove(file_path)
                            raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_SIZE_MB}MB")
                        buffer.write(chunk)
            finally:
                try:
                    await file.seek(0)
                except Exception:
                    pass

            saved_files.append(file_path)
    except Exception:
        # Clean temp folder on any error
        cleanup_folder(upload_dir)
        raise
    
    extraction_config = PDFExtractionConfig(
        save_to_files=False
    )
    
    # Create and run enhanced pipeline via use case
    ingestion_service = PipelineIngestionService()
    use_case = UploadDocumentsUseCase(UploadDeps(ingestion=ingestion_service))
    results_summary = await use_case.run(
        documents_folder=upload_dir,
        output_folder=os.path.join(upload_dir, "output"),
        config={
            **(config.__dict__ if hasattr(config, "__dict__") else {}),
            "enable_ocr": getattr(extraction_config, "enable_ocr", True),
            "enable_vlm": getattr(extraction_config, "enable_vlm", True),
            "enable_table_extraction": getattr(extraction_config, "enable_table_extraction", True),
            "enable_image_extraction": getattr(extraction_config, "enable_image_extraction", True),
            "save_to_files": getattr(extraction_config, "save_to_files", False),
            "output_format": getattr(extraction_config, "output_format", "markdown"),
        },
        clean_before_ingest=False,
    )

    # Clean temp folder after processing
    cleanup_folder(upload_dir)

    return {
        "success": True,
        "saved_files": [os.path.basename(p) for p in saved_files],
        "config_used": config.__dict__ if config else None,
        "processed_documents": results_summary.get("processed", 0),
        "ingest_success": results_summary.get("success", 0),
        "cleaned": True
    }


@app.get("/users/{user_id}/sessions", response_model=SessionListResponse)
async def get_user_sessions(user_id: str):
    """Get all sessions for a user."""
    try:
        sessions_repo = AsyncpgSessionRepository()
        
        # Get all sessions for user using your existing function
        sessions = await sessions_repo.get_user_sessions(user_id)
        
        session_list = [
            SessionInfo(
                session_id=s["session_id"],
                user_id=s["user_id"],
                created_at=s["created_at"],
                updated_at=s["updated_at"],
                message_count=s.get("message_count", 0),
                last_message=s.get("last_message", "")[:100] if s.get("last_message") else None
            )
            for s in sessions
        ]
        
        return SessionListResponse(
            sessions=session_list,
            total=len(session_list)
        )
    
    except Exception as e:
        logger.error(f"Failed to get user sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/users/{user_id}/sessions")
async def create_new_session(user_id: str):
    """Create a new session for user."""
    try:
        sessions_repo = AsyncpgSessionRepository()
        
        # Create session using your existing function
        session_id = await sessions_repo.create(
            user_id=user_id,
            metadata={}
        )
        
        return {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its messages."""
    try:
        sessions_repo = AsyncpgSessionRepository()
        messages_repo = AsyncpgMessageRepository()
        
        # First delete all messages in the session
        await messages_repo.delete(session_id)
        
        # Delete session 
        await sessions_repo.delete(session_id)
        
        return {"success": True, "message": "Session deleted"}
    
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/exists")
async def check_user_exists(user_id: str):
    """Check if user has any sessions."""
    try:
        sessions_repo = AsyncpgSessionRepository()
        
        # Check if user exists using your new function
        exists = await sessions_repo.check_user_exists(user_id)
        
        # Get session count
        if exists:
            sessions = await sessions_repo.get_user_sessions(user_id)
            session_count = len(sessions)
        else:
            session_count = 0
        
        return {
            "exists": exists,
            "user_id": user_id,
            "session_count": session_count
        }
    
    except Exception as e:
        logger.error(f"Failed to check user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/messages", response_model=MessageHistoryResponse)
async def get_session_messages(session_id: str, limit: Optional[int] = None):
    """Get all messages for a session."""
    try:
        messages_repo = AsyncpgMessageRepository()
        
        # Get messages using your existing function
        messages = await messages_repo.list(session_id, limit=limit)
        
        message_list = [
            MessageInfo(
                id=m["id"],
                role=m["role"],
                content=m["content"],
                created_at=m["created_at"],
                metadata=m.get("metadata", {})
            )
            for m in messages
        ]
        
        return MessageHistoryResponse(
            messages=message_list,
            session_id=session_id,
            total=len(message_list)
        )
    
    except Exception as e:
        logger.error(f"Failed to get session messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")

    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=str(exc),
            error_type=type(exc).__name__,
            request_id=str(uuid.uuid4()),
        ).model_dump(),
    )


# Run the app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(
        "src.agent_tiers.infrastructure.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_ENV == "development",
        log_level=API_LOG_LEVEL.lower(),
    )
