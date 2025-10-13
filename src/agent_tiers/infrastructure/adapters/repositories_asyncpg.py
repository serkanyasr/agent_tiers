from typing import Optional, Dict, Any, List

from ...domain.ports.repositories import SessionRepository, MessageRepository, DocumentRepository

from ...infrastructure.db.rag_db import (
    create_session as db_create_session,
    get_session as db_get_session,
    add_message as db_add_message,
    get_session_messages as db_get_session_messages,
    get_document as db_get_document,
    list_documents as db_list_documents,
    delete_session as db_delete_session,
    delete_message as db_delete_message,
    get_user_sessions as db_get_user_sessions,
    check_user_exists as db_check_user_exists,
    db_pool,
)


class AsyncpgSessionRepository(SessionRepository):
    """Session repository using existing memory_db functions."""
    
    async def create(self, user_id: Optional[str], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create new session."""
        return await db_create_session(user_id=user_id, metadata=metadata)

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        return await db_get_session(session_id)
    
    async def delete(self, session_id: str) -> None:
        """Delete session."""
        await db_delete_session(session_id)
    
    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for a user."""
        return await db_get_user_sessions(user_id)
    
    async def check_user_exists(self, user_id: str) -> bool:
        """Check if user has any sessions."""
        return await db_check_user_exists(user_id)



class AsyncpgMessageRepository(MessageRepository):
    async def add(self, session_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add message to session."""
        return await db_add_message(session_id=session_id, role=role, content=content, metadata=metadata or {})

    async def list(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List messages in a session."""
        return await db_get_session_messages(session_id, limit=limit)

    async def delete(self, session_id: str) -> None:
        """Delete all messages in a session."""
        await db_delete_message(session_id)
        

class AsyncpgDocumentRepository(DocumentRepository):
    async def get(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        return await db_get_document(db_pool.pool, document_id)

    async def list(self, limit: int = 100, offset: int = 0, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List documents with optional filtering."""
        return await db_list_documents(db_pool.pool, limit=limit, offset=offset, metadata_filter=metadata_filter)


