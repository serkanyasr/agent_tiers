from mem0 import Memory
from dataclasses import dataclass
from dotenv import load_dotenv
import psycopg2
load_dotenv()

import logging
import os

logger = logging.getLogger(__name__)


def get_mem0_client() -> Memory:
    
    try:
        config = {
            "llm": {
                "provider": os.getenv("MEMORY_LLM_PROVIDER", "openai"),
                "config": {
                    "model": os.getenv("MEMORY_LLM_MODEL", "gpt-4o-mini"),
                    "api_key": os.getenv("OPENAI_API_KEY", "LLM_API_KEY"),
                },
            },
            "embedder": {
                "provider": os.getenv("MEMORY_EMBEDDING_PROVIDER", "openai"),
                "config": {
                    "model": os.getenv(
                        "MEMORY_EMBEDDING_MODEL", "text-embedding-3-small"
                    ),
                    "embedding_dims": int(os.getenv("MEMORY_EMBEDDING_DIMS", 1536)),
                    "api_key": os.getenv("OPENAI_API_KEY", "LLM_API_KEY"),
                },
            },
            "vector_store": {
                "provider": os.getenv("MEMORY_VECTOR_STORE_PROVIDER", "pgvector"),
                "config": {
                    "user": os.getenv("MEMORY_DB_USER", "postgres"),
                    "password": os.getenv("MEMORY_DB_PASSWORD", "postgres"),
                    "host": os.getenv("MEMORY_DB_HOST", "localhost"),
                    "port": int(os.getenv("MEMORY_DB_PORT", 6543)),
                    "dbname": os.getenv("MEMORY_DB_NAME", "memories"),
                },
            },


        }
        _memory = Memory.from_config(config)
        
        logger.info("Memory has been configured and started")
        return _memory
    except Exception as e:
        logger.error(f"Memory startup failed: {e}")
        raise


def test_db_connection() -> bool:
    """
    Test the connection to the PostgreSQL database and check if the pgvector extension is enabled.
    """
    try:
        # Establish database connection using environment variables (with defaults)
        conn = psycopg2.connect(
            host=os.getenv("MEMORY_DB_HOST", "localhost"),
            port=int(os.getenv("MEMORY_DB_PORT", 6543)),
            user=os.getenv("MEMORY_DB_USER", "postgres"),
            password=os.getenv("MEMORY_DB_PASSWORD", "postgres"),
            database=os.getenv("MEMORY_DB_NAME", "memories")
        )
        
        with conn.cursor() as cursor:
            # Check PostgreSQL version
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            print(f"PostgreSQL version: {version[0]}")
            
            # Check if pgvector extension is enabled
            cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
            vector_ext = cursor.fetchone()
            if vector_ext:
                print("pgvector extension is enabled.")
            else:
                print("pgvector extension is NOT enabled! ")
                
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
                logger.info("pgvector extension is initialized.")
                
        conn.close()
        return True
    except Exception as e:
        print(f"Database connection error: {e}")
        return False

    
@dataclass
class Mem0Context:
    """Context for the Mem0 MCP server."""
    mem0_client: Memory
    
    