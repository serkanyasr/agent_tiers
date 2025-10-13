from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import ClassVar

load_dotenv()


class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"

    # Embeddings
    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMS: int = 1536

    # Memory Vector Store (Postgres + pgvector)
    MEMORY_VECTOR_STORE_PROVIDER: str = "pgvector"
    MEMORY_DB_USER: str
    MEMORY_DB_PASSWORD: str
    MEMORY_DB_HOST: str
    MEMORY_DB_PORT: int
    MEMORY_DB_NAME: str
    MEMORY_LLM_PROVIDER: str = "openai"
    MEMORY_LLM_MODEL: str = "gpt-4o-mini"
    MEMORY_EMBEDDING_PROVIDER: str = "openai"
    MEMORY_EMBEDDING_MODEL: str = "text-embedding-3-small"
    MEMORY_EMBEDDING_DIMS: int = 1536
    MEMORY_MCP_TRANSPORT: str = "streamable-http"

    # RAG Vector Store (Postgres + pgvector)
    RAG_MCP_TRANSPORT: str = "streamable-http"
    RAG_DB_USER: str
    RAG_DB_PASSWORD: str
    RAG_DB_HOST: str
    RAG_DB_PORT: int
    RAG_DB_NAME: str

    # API
    API_ENV: str
    API_HOST: str
    API_PORT: int
    API_LOG_LEVEL: str

    # UI
    APP_HOST: str
    APP_PORT: int
    CHAINLIT_AUTH_SECRET: str

    # Upload constraints
    UPLOAD_MAX_FILE_MB: int = 25
    UPLOAD_ALLOWED_EXTENSIONS: str = ".pdf,.docx,.doc,.txt,.md,.png,.jpg,.jpeg"

    BASE_DIR: ClassVar[str] = Path(__file__).resolve().parents[1]
    SCHEMA_PATH: ClassVar[str] = BASE_DIR / "infrastructure" / "sql" / "schema.sql"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
