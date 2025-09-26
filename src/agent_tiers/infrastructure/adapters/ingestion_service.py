from typing import Dict, Any, List

from src.agent_tiers.domain.ports.services import IngestionService
from src.agent_tiers.infrastructure.ingestion.ingest import (
    DocumentIngestionPipeline,
    IngestionConfig,
    PDFExtractionConfig,
    progress_callback,
)


class PipelineIngestionService(IngestionService):
    async def ingest_documents(
        self,
        documents_folder: str,
        output_folder: str,
        config: Dict[str, Any],
        clean_before_ingest: bool = False,
    ) -> List[Dict[str, Any]]:
        ingestion_config = IngestionConfig(
            chunk_size=config.get("chunk_size", 850),
            chunk_overlap=config.get("chunk_overlap", 150),
            use_semantic_chunking=config.get("use_semantic_chunking", True),
        )

        extraction_config = PDFExtractionConfig(
            enable_ocr=config.get("enable_ocr", True),
            enable_vlm=config.get("enable_vlm", True),
            enable_table_extraction=config.get("enable_table_extraction", True),
            enable_image_extraction=config.get("enable_image_extraction", True),
            save_to_files=config.get("save_to_files", True),
            output_format=config.get("output_format", "markdown"),
            output_directory=output_folder,
        )

        pipeline = DocumentIngestionPipeline(
            config=ingestion_config,
            documents_folder=documents_folder,
            output_folder=output_folder,
            clean_before_ingest=clean_before_ingest,
            extraction_config=extraction_config,
        )

        results = await pipeline.ingest_documents(progress_callback)

        # Normalize results to dicts
        return [r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in results]


