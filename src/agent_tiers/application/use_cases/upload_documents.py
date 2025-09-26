from typing import  Dict, Any
from dataclasses import dataclass

from src.agent_tiers.domain.ports.services import IngestionService


@dataclass
class UploadDeps:
    """Dependencies for upload documents use case."""
    ingestion: IngestionService


class UploadDocumentsUseCase:
    """Use case for uploading and ingesting documents."""
    
    def __init__(self, deps: UploadDeps):
        """Initialize upload documents use case with dependencies."""
        self.deps = deps

    async def run(self, documents_folder: str, output_folder: str, config: Dict[str, Any], clean_before_ingest: bool = False) -> Dict[str, Any]:
        """Execute document upload and ingestion process."""
        results = await self.deps.ingestion.ingest_documents(
            documents_folder=documents_folder,
            output_folder=output_folder,
            config=config,
            clean_before_ingest=clean_before_ingest,
        )
        total = len(results)
        success = len([r for r in results if r.get("chunks_created", 0) > 0])
        return {"processed": total, "success": success, "results": results}


