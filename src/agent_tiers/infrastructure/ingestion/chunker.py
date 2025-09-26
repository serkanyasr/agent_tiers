
import logging
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

@dataclass
class ChunkingConfig:
    """Simple configuration for PDF chunking."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000

    use_semantic_splitting: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        if self.min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be positive")
        
@dataclass
class DocumentChunk:
    """Represents a document chunk."""
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    token_count: Optional[int] = None
    
    def __post_init__(self):
        """Calculate token count if not provided."""
        if self.token_count is None:
            self.token_count = len(self.content) // 4
            
class DocSemanticChunker:
    """ Semantic chunker for PDF documents."""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Semantic splitter
        if config.use_semantic_splitting:
            self.semantic_splitter = SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="percentile" # Split %10 diff.
            )
        
        # Recursive splitter
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
        )
    
    def chunk_content(
        self,
        content: str,
        title: str = "Document",
        source: str = "pdf",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Chunk content into semantic pieces and convert to DocumentChunk objects.
        """
        if not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "content_type": "pdf",
            **(metadata or {})
        }

        doc = Document(page_content=content, metadata=base_metadata)

        try:
            if self.config.use_semantic_splitting and len(content) > self.config.chunk_size:
                chunks = self.semantic_splitter.split_documents([doc])
                for chunk in chunks:
                    chunk.metadata["chunk_method"] = "semantic"
            else:
                chunks = self.fallback_splitter.split_documents([doc])
                for chunk in chunks:
                    chunk.metadata["chunk_method"] = "recursive"

        except Exception as e:
            logger.warning(f"Semantic chunking failed, using fallback: {e}")
            chunks = self.fallback_splitter.split_documents([doc])
            for chunk in chunks:
                chunk.metadata["chunk_method"] = "fallback"

        # Filter small chunks and convert to DocumentChunk
        final_chunks = []
        for i, chunk in enumerate(chunks):
            text = chunk.page_content.strip()
            if len(text) >= self.config.min_chunk_size:
                chunk.metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(text)
                })
                final_chunks.append(
                    DocumentChunk(
                        content=text,
                        index=i,
                        start_char=0,  
                        end_char=len(text),
                        metadata=chunk.metadata
                    )
                )

        return final_chunks

def create_chunker(config: ChunkingConfig) -> DocSemanticChunker:
    """Create  chunker with simple configuration."""
    return DocSemanticChunker(config)