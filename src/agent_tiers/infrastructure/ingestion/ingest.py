import argparse
import asyncio
from datetime import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field, field_validator

from ..db.rag_db import close_database, initialize_database, db_pool, execute_init_sql
from .extract import create_pdf_extractor, PDFExtractionConfig, DocumentType
from .chunker import ChunkingConfig, DocumentChunk, create_chunker
from ..llm_providers.providers import get_openai_embedding_model
from ..config import settings
from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class IngestionConfig(BaseModel):
    """Configuration for document ingestion."""
    chunk_size: int = Field(default=850, ge=100, le=5000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    max_chunk_size: int = Field(default=2000, ge=500, le=10000)
    use_semantic_chunking: bool = True

    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        chunk_size = info.data.get('chunk_size', 1000)
        if v >= chunk_size:
            raise ValueError(f"Chunk overlap ({v}) must be less than chunk size ({chunk_size})")
        return v


class IngestionResult(BaseModel):
    """Result of document ingestion."""
    document_id: str
    title: str
    document_type: str
    pipeline_used: str
    chunks_created: int
    processing_time_ms: float
    output_file: Optional[str] = None
    metadata: Dict[str, Any] = {}


class DocumentIngestionPipeline:
    """Enhanced pipeline for ingesting documents with Docling smart processing."""

    def __init__(self, 
                 config: IngestionConfig, 
                 documents_folder: str = "documents",
                 output_folder: str = "processed_documents",
                 clean_before_ingest: bool = False, 
                 sql_schema_path: str = "sql/schema.sql",
                 extraction_config: Optional[PDFExtractionConfig] = None):
        """
        Initialize enhanced ingestion pipeline with Docling.
        
        Args:
            config: Ingestion configuration
            documents_folder: Folder containing documents to process
            output_folder: Folder to save processed documents  
            clean_before_ingest: Whether to clean existing data before ingestion
            sql_schema_path: Path to SQL schema file
            extraction_config: Optional custom extraction configuration
        """
        
        self.config = config
        self.documents_folder = documents_folder
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.clean_before_ingest = clean_before_ingest
        self.sql_schema_path = sql_schema_path
        
        if not sql_schema_path:
            self.sql_schema_path = settings.SCHEMA_PATH
            
        # Setup extraction configuration
        if extraction_config is None:
            extraction_config = PDFExtractionConfig(
                enable_ocr=True,
                enable_vlm=True,
                enable_table_extraction=True,
                enable_image_extraction=True,
                save_to_files=True,
                output_format="markdown",
                output_directory=str(self.output_folder)
            )
        
        # Create PDF extractor with Docling
        self.extractor = create_pdf_extractor(extraction_config)
        
        # Configure chunking
        self.chunker_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_chunk_size=config.max_chunk_size,
            use_semantic_splitting=config.use_semantic_chunking
        )

        # Create chunker
        self.chunker = create_chunker(self.chunker_config)
        self._initialized = False
        
    async def initialize(self):
        """Initialize database connections."""
        if self._initialized:
            return
        
        logger.info("Initializing document ingestion pipeline...")
        
        # Initialize database connections
        await initialize_database()
        await execute_init_sql(self.sql_schema_path)

        self._initialized = True
        logger.info("Document ingestion pipeline initialized")
    
    async def close(self):
        """Close database connections."""
        if self._initialized:
            await close_database()
            self._initialized = False
        
    async def _clean_databases(self):
        """Clean existing data from databases."""
        logger.warning("Cleaning existing data from databases...")
        
        # Clean PostgreSQL
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM messages")
                await conn.execute("DELETE FROM sessions") 
                await conn.execute("DELETE FROM chunks")
                await conn.execute("DELETE FROM documents")
        
        logger.info("Cleaned PostgreSQL database")

    async def _ingest_single_document(self, file_path: str) -> IngestionResult:
        """
        Ingest a single document with Docling smart processing.
        
        Args:
            file_path: Path to the document file
        
        Returns:
            Ingestion result
        """
        start_time = datetime.now()
        
        try:
            # Extract document content using Docling
            extraction_result = self.extractor.extract_pdf_content(file_path)
            
            if not extraction_result.success:
                logger.error(f"Failed to extract content from {file_path}: {extraction_result.error_message}")
                return IngestionResult(
                    document_id="",
                    title=os.path.basename(file_path),
                    document_type=extraction_result.document_type.value,
                    pipeline_used=extraction_result.pipeline_used,
                    chunks_created=0,
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    metadata=extraction_result.metadata
                )
            
            document_content = extraction_result.content
            document_metadata = extraction_result.metadata
            document_source = os.path.relpath(file_path, self.documents_folder)
            
            # Determine document title
            document_title = document_metadata.get("title", "")
            if not document_title or document_title.strip() == "":
                document_title = Path(file_path).stem

            logger.info(f"Processing document: {document_title}")
            logger.info(f"Document type: {extraction_result.document_type.value}")
            logger.info(f"Pipeline used: {extraction_result.pipeline_used}")
            logger.info(f"Content length: {len(document_content)} characters")

            # Chunk the document content
            main_chunks = self.chunker.chunk_content(
                content=document_content,
                title=document_title,
                source=document_source,
                metadata=document_metadata
            )
            
            if not main_chunks:
                logger.warning(f"No chunks created for {document_title}")
                return IngestionResult(
                    document_id="",
                    title=document_title,
                    document_type=extraction_result.document_type.value,
                    pipeline_used=extraction_result.pipeline_used,
                    chunks_created=0,
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    output_file=extraction_result.output_file_path,
                    metadata=document_metadata
                )

            logger.info(f"Created {len(main_chunks)} chunks")
            
            # Generate embeddings for all chunks
            embedded_chunks = await self.aembed_chunks(
                chunks=main_chunks,
                model=get_openai_embedding_model()
            )
            logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
            
            # Save to PostgreSQL
            document_id = await self._save_to_postgres(
                document_title,
                document_source,
                document_content,
                embedded_chunks,
                document_metadata
            )
            
            logger.info(f"Saved document to PostgreSQL with ID: {document_id}")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return IngestionResult(
                document_id=document_id,
                title=document_title,
                document_type=extraction_result.document_type.value,
                pipeline_used=extraction_result.pipeline_used,
                chunks_created=len(main_chunks),
                processing_time_ms=processing_time,
                output_file=extraction_result.output_file_path,
                metadata=document_metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            return IngestionResult(
                document_id="",
                title=os.path.basename(file_path),
                document_type=DocumentType.UNKNOWN.value,
                pipeline_used="error",
                chunks_created=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                metadata={"error": str(e)}
            )
    
    async def ingest_documents(self, progress_callback: Optional[callable] = None) -> List[IngestionResult]:
        """
        Ingest all documents from the documents folder with smart Docling processing.
        
        Args:
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of ingestion results
        """
        if not self._initialized:
            await self.initialize()
        
        # Clean existing data if requested
        if self.clean_before_ingest:
            await self._clean_databases()

        # Find all supported files
        supported_files = self._find_supported_files(self.documents_folder)

        if not supported_files:
            logger.warning(f"No supported files found in {self.documents_folder}")
            return []

        logger.info(f"Found {len(supported_files)} supported files to process")

        results = []

        for i, file_path in enumerate(supported_files):
            try:
                logger.info(f"Processing file {i+1}/{len(supported_files)}: {file_path}")

                result = await self._ingest_single_document(file_path)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(supported_files))
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append(IngestionResult(
                    document_id="",
                    title=os.path.basename(file_path),
                    document_type=DocumentType.UNKNOWN.value,
                    pipeline_used="error",
                    chunks_created=0,
                    processing_time_ms=0,
                    metadata={"error": str(e)}
                ))
        
        # Log summary
        total_chunks = sum(r.chunks_created for r in results)
        successful_results = [r for r in results if r.chunks_created > 0]

        logger.info(f"Document ingestion complete: {len(results)} documents, {total_chunks} chunks")
        logger.info(f"Successful: {len(successful_results)}, Failed: {len(results) - len(successful_results)}")

        return results

    async def aembed_chunks(self, chunks: List[DocumentChunk], model: str = "text-embedding-3-small") -> List[DocumentChunk]:
        """Generate embeddings for chunks (LangChain handles batching internally)."""
        embeddings = OpenAIEmbeddings(model=model)

        # Get all chunk contents
        texts = [chunk.content for chunk in chunks]

        vectors = await embeddings.aembed_documents(texts)

        embedded_chunks = []
        for chunk, vector in zip(chunks, vectors):
            embedded_chunk = DocumentChunk(
                content=chunk.content,
                index=chunk.index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata={
                    **chunk.metadata,
                    "embedding_model": model,
                    "embedding_generated_at": datetime.now().isoformat(),
                },
            )
            embedded_chunk.embedding = vector
            embedded_chunks.append(embedded_chunk)

        return embedded_chunks

    def _find_supported_files(self, directory: str, recursive: bool = True) -> List[str]:
        """
        Find all supported files in a directory.

        Args:
            directory: Directory path to search
            recursive: Whether to search subdirectories

        Returns:
            List of supported file paths
        """
        directory_path = Path(directory)

        if not directory_path.exists() or not directory_path.is_dir():
            raise FileNotFoundError(f"Directory not found or not a directory: {directory_path}")

        # Get supported extensions from extractor
        supported_extensions = self.extractor.get_supported_extensions()

        if recursive:
            all_files = list(directory_path.rglob("*"))
        else:
            all_files = list(directory_path.glob("*"))

        supported_files = [
            str(file.resolve()) 
            for file in all_files 
            if file.is_file() and file.suffix.lower() in supported_extensions
        ]
        
        logger.info(f"Found {len(supported_files)} supported files in {directory_path}")
        return supported_files

    async def _save_to_postgres(
        self,
        title: str,
        source: str,
        content: str,
        chunks: List[DocumentChunk],
        metadata: Dict[str, Any]
    ) -> str:
        """Save document and chunks to PostgreSQL with enhanced metadata."""
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                # Insert document
                document_result = await conn.fetchrow(
                    """
                    INSERT INTO documents (title, source, content, metadata)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id::text
                    """,
                    title,
                    source,
                    content,
                    json.dumps(metadata)
                )
                
                document_id = document_result["id"]
                
                # Insert chunks with enhanced metadata
                for chunk in chunks:
                    embedding_data = None
                    if hasattr(chunk, 'embedding') and chunk.embedding:
                        # PostgreSQL vector format
                        embedding_data = '[' + ','.join(map(str, chunk.embedding)) + ']'

                    # Enhanced metadata for chunks
                    chunk_metadata = {
                        **chunk.metadata,
                        "chunk_type": chunk.metadata.get("content_type", "text"),
                        "document_type": metadata.get("document_type", "unknown"),
                        "pipeline_used": metadata.get("pipeline_used", "unknown")
                    }

                    await conn.execute(
                        """
                        INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata, token_count)
                        VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
                        """,
                        document_id,
                        chunk.content,
                        embedding_data,
                        chunk.index,
                        json.dumps(chunk_metadata),
                        chunk.token_count if hasattr(chunk, 'token_count') else len(chunk.content.split())
                    )
                
                return document_id


def progress_callback(current: int, total: int):
    print(f"Progress: {current}/{total} documents processed")

async def main():
    """Main function for running enhanced document ingestion with Docling."""
    parser = argparse.ArgumentParser(description="Enhanced Document ingestion with Docling smart processing")
    parser.add_argument("--documents", "-d", default="documents", help="Documents folder path")
    parser.add_argument("--output", "-o", default="processed_documents", help="Output folder for processed documents")
    parser.add_argument("--clean", "-c", action="store_true", help="Clean existing data before ingestion")
    parser.add_argument("--chunk-size", type=int, default=850, help="Chunk size for splitting documents")
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic chunking")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap size")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--sql-schema-path", "-sql", default="sql/schema.sql", help="Path to SQL schema file")
    
    # Docling-specific extraction options
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR processing")
    parser.add_argument("--no-vlm", action="store_true", help="Disable VLM processing")
    parser.add_argument("--no-tables", action="store_true", help="Skip table extraction")
    parser.add_argument("--no-images", action="store_true", help="Skip image extraction")
    parser.add_argument("--no-file-output", action="store_true", help="Skip saving to output files")
    parser.add_argument("--output-format", choices=["markdown", "json", "text"], default="markdown", 
                       help="Output format for saved files")
    parser.add_argument("--images-scale", type=float, default=1.0, help="Image scaling factor")
    parser.add_argument("--force-ocr", action="store_true", help="Force full page OCR")
    
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create ingestion configuration
    config = IngestionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_semantic_chunking=not args.no_semantic,
    )
    
    # Create extraction configuration
    extraction_config = PDFExtractionConfig(
        enable_ocr=not args.no_ocr,
        enable_vlm=not args.no_vlm,
        enable_table_extraction=not args.no_tables,
        enable_image_extraction=not args.no_images,
        save_to_files=not args.no_file_output,
        output_format=args.output_format,
        output_directory=args.output,
        images_scale=args.images_scale,
        force_full_page_ocr=args.force_ocr,
        generate_page_images=args.force_ocr or not args.no_ocr
    )
    
    # Create and run enhanced pipeline
    pipeline = DocumentIngestionPipeline(
        config=config,
        documents_folder=args.documents,
        output_folder=args.output,
        clean_before_ingest=args.clean,
        sql_schema_path=args.sql_schema_path,
        extraction_config=extraction_config
    )

    try:
        start_time = datetime.now()
        
        results = await pipeline.ingest_documents(progress_callback)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Print enhanced summary
        print("\n" + "="*70)
        print("DOCLING INGESTION SUMMARY")
        print("="*70)
        print(f"Documents processed: {len(results)}")
        print(f"Total chunks created: {sum(r.chunks_created for r in results)}")
        print(f"Successful documents: {len([r for r in results if r.chunks_created > 0])}")
        print(f"Failed documents: {len([r for r in results if r.chunks_created == 0])}")
        print(f"Output folder: {args.output}")
        print(f"File output enabled: {not args.no_file_output}")
        print(f"Output format: {args.output_format}")
        print(f"OCR enabled: {not args.no_ocr}")
        print(f"VLM enabled: {not args.no_vlm}")
        print(f"Table extraction: {not args.no_tables}")
        print(f"Image extraction: {not args.no_images}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print("="*70)
        
        # Print pipeline usage statistics
        pipeline_stats = {}
        doc_type_stats = {}
        
        for result in results:
            if result.chunks_created > 0:
                # Pipeline statistics
                pipeline = result.pipeline_used
                pipeline_stats[pipeline] = pipeline_stats.get(pipeline, 0) + 1
                
                # Document type statistics
                doc_type = result.document_type
                doc_type_stats[doc_type] = doc_type_stats.get(doc_type, 0) + 1
        
        if pipeline_stats:
            print("\nPIPELINE USAGE:")
            for pipeline, count in sorted(pipeline_stats.items()):
                print(f"  {pipeline}: {count} documents")
        
        if doc_type_stats:
            print("\nDOCUMENT TYPES:")
            for doc_type, count in sorted(doc_type_stats.items()):
                print(f"  {doc_type}: {count} documents")
        
        # Print per-document summary
        print("\nPER-DOCUMENT RESULTS:")
        for result in results:
            status = "✅" if result.chunks_created > 0 else "❌"
            pipeline_info = f"({result.pipeline_used})" if result.chunks_created > 0 else "(failed)"
            output_info = f" → {result.output_file}" if result.output_file else ""
            print(f"  {status} {result.title}: {result.chunks_created} chunks {pipeline_info} "
                f"({result.processing_time_ms/1000:.1f}s){output_info}")

        # Print extraction statistics
        extraction_stats = {}
        for result in results:
            if result.chunks_created > 0:
                doc_type = result.document_type
                if doc_type not in extraction_stats:
                    extraction_stats[doc_type] = {
                        'count': 0,
                        'total_chunks': 0,
                        'avg_processing_time': 0
                    }
                extraction_stats[doc_type]['count'] += 1
                extraction_stats[doc_type]['total_chunks'] += result.chunks_created
                extraction_stats[doc_type]['avg_processing_time'] += result.processing_time_ms
        
        if extraction_stats:
            print("\nEXTRACTION STATISTICS:")
            for doc_type, stats in extraction_stats.items():
                avg_time = stats['avg_processing_time'] / stats['count'] / 1000
                avg_chunks = stats['total_chunks'] / stats['count']
                print(f"  {doc_type}: {stats['count']} docs, avg {avg_chunks:.1f} chunks, avg {avg_time:.1f}s")

    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user")
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        raise
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())