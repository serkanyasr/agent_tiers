from typing import Optional, Dict, Any, List

from src.agent_tiers.domain.ports.repositories import SessionRepository, MessageRepository, DocumentRepository
from src.agent_tiers.infrastructure.db.rag_db import (
    create_session as db_create_session,
    get_session as db_get_session,
    add_message as db_add_message,
    get_session_messages as db_get_session_messages,
    get_document as db_get_document,
    list_documents as db_list_documents,
    db_pool,
)


class AsyncpgSessionRepository(SessionRepository):
    async def create(self, user_id: Optional[str], metadata: Optional[Dict[str, Any]]) -> str:
        return await db_create_session(user_id=user_id, metadata=metadata)

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        return await db_get_session(session_id)


class AsyncpgMessageRepository(MessageRepository):
    async def add(self, session_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        return await db_add_message(session_id=session_id, role=role, content=content, metadata=metadata or {})

    async def list(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return await db_get_session_messages(session_id, limit=limit)


class AsyncpgDocumentRepository(DocumentRepository):
    async def get(self, document_id: str) -> Optional[Dict[str, Any]]:
        # db_pool.pool kullanımı gereken imzayı talep ediyor
        return await db_get_document(db_pool.pool, document_id)

    async def list(self, limit: int = 100, offset: int = 0, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return await db_list_documents(db_pool.pool, limit=limit, offset=offset, metadata_filter=metadata_filter)


