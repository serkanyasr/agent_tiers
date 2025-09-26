import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

# Docling imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    VlmPipelineOptions, 
    AsrPipelineOptions
)
from docling.document_converter import (
    DocumentConverter, 
    PdfFormatOption,
    FormatOption
)
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.pipeline.vlm_pipeline import VlmPipeline

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Document type definitions for smart processing."""
    PDF_TEXT = "pdf_text"
    PDF_SCANNED = "pdf_scanned"
    PDF_MULTIMODAL = "pdf_multimodal"
    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO_VIDEO = "audio_video"
    UNKNOWN = "unknown"


@dataclass
class ExtractionResult:
    """Result of document extraction."""
    success: bool
    content: str
    document_type: DocumentType
    pipeline_used: str
    metadata: Dict[str, Any]
    output_file_path: Optional[str] = None
    error_message: Optional[str] = None


class PDFExtractionConfig(BaseModel):
    """Configuration for PDF extraction using Docling."""
    
    # Docling-specific options
    enable_ocr: bool = Field(default=True, description="Enable OCR for scanned documents")
    enable_vlm: bool = Field(default=True, description="Enable Vision-Language Model processing")
    enable_table_extraction: bool = Field(default=True, description="Extract tables from documents")
    enable_image_extraction: bool = Field(default=True, description="Extract images from documents")
    images_scale: float = Field(default=1.0, ge=0.1, le=5.0, description="Image scaling factor")
    
    # Output options
    save_to_files: bool = Field(default=True, description="Save processed documents to files")
    output_format: str = Field(default="markdown", description="Output format: markdown, json, text")
    output_directory: Optional[str] = Field(default=None, description="Output directory for processed files")
    
    # Processing options
    force_full_page_ocr: bool = Field(default=False, description="Force OCR on entire pages")
    generate_page_images: bool = Field(default=False, description="Generate page images during processing")


class DoclingExtractor:
    """Smart document extractor using Docling with automatic pipeline selection."""
    
    def __init__(self, config: PDFExtractionConfig):
        """
        Initialize Docling extractor.
        
        Args:
            config: Extraction configuration
        """
        self.config = config
        
        # Setup output directory if file saving is enabled
        if config.save_to_files and config.output_directory:
            self.output_dir = Path(config.output_directory)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
            
        # Setup Docling pipelines
        self._setup_docling_pipelines()
        
        # File type mapping for document type detection
        self.file_type_mapping = {
            '.pdf': self._analyze_pdf_type,
            '.png': lambda x: DocumentType.IMAGE,
            '.jpg': lambda x: DocumentType.IMAGE,
            '.jpeg': lambda x: DocumentType.IMAGE,
            '.tiff': lambda x: DocumentType.IMAGE,
            '.bmp': lambda x: DocumentType.IMAGE,
            '.webp': lambda x: DocumentType.IMAGE,
            '.docx': lambda x: DocumentType.DOCUMENT,
            '.doc': lambda x: DocumentType.DOCUMENT,
            '.xlsx': lambda x: DocumentType.DOCUMENT,
            '.pptx': lambda x: DocumentType.DOCUMENT,
            '.txt': lambda x: DocumentType.DOCUMENT,
            '.md': lambda x: DocumentType.DOCUMENT,
            '.mp4': lambda x: DocumentType.AUDIO_VIDEO,
            '.mp3': lambda x: DocumentType.AUDIO_VIDEO,
            '.wav': lambda x: DocumentType.AUDIO_VIDEO,
            '.avi': lambda x: DocumentType.AUDIO_VIDEO,
            '.mov': lambda x: DocumentType.AUDIO_VIDEO,
        }

    def _setup_docling_pipelines(self):
        """Setup different Docling pipelines for different document types."""
        
        # Standard PDF Pipeline Options (for text-based PDFs)
        self.pdf_text_options = PdfPipelineOptions(
            do_ocr=False,  # Text-based PDFs don't need OCR
            do_picture_description=False,
            do_table_structure=self.config.enable_table_extraction,
            generate_page_images=self.config.generate_page_images,
            generate_picture_images=self.config.enable_image_extraction,
            images_scale=self.config.images_scale
        )
        
        # OCR-enabled PDF Pipeline Options (for scanned PDFs)
        self.pdf_scanned_options = PdfPipelineOptions(
            do_ocr=self.config.enable_ocr,
            do_picture_description=False,
            do_table_structure=self.config.enable_table_extraction,
            generate_page_images=True,  # Always generate for scanned docs
            generate_picture_images=self.config.enable_image_extraction,
            force_full_page_ocr=self.config.force_full_page_ocr,
            images_scale=self.config.images_scale
        )
        
        # VLM Pipeline Options (for multimodal content)
        self.vlm_options = VlmPipelineOptions(
            generate_page_images=True,  # VLM needs images
        )
        
        # ASR Pipeline Options (for audio/video)
        self.asr_options = AsrPipelineOptions()

    def _analyze_pdf_type(self, file_path: str) -> DocumentType:
        """
        Analyze PDF type using simple heuristics.
        In production, this could be enhanced with actual content analysis.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Detected PDF type
        """
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            # Simple heuristic based on file size
            # Large files often contain images/scans
            if file_size_mb > 20:
                return DocumentType.PDF_MULTIMODAL
            elif file_size_mb > 5:
                return DocumentType.PDF_SCANNED
            else:
                return DocumentType.PDF_TEXT
                
        except Exception as e:
            logger.warning(f"Could not analyze PDF type for {file_path}: {e}")
            return DocumentType.PDF_TEXT  # Safe default

    def detect_document_type(self, file_path: str) -> DocumentType:
        """
        Detect document type for smart pipeline selection.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Detected document type
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension in self.file_type_mapping:
            detector = self.file_type_mapping[extension]
            return detector(str(file_path))
        else:
            logger.warning(f"Unknown file extension: {extension}")
            return DocumentType.UNKNOWN

    def _get_docling_converter(self, doc_type: DocumentType) -> Tuple[DocumentConverter, str]:
        """
        Get appropriate Docling converter for document type.
        
        Args:
            doc_type: Detected document type
            
        Returns:
            Tuple of (converter, pipeline_name)
        """
        
        if doc_type == DocumentType.PDF_TEXT:
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=StandardPdfPipeline,
                        pipeline_options=self.pdf_text_options
                    )
                }
            )
            return converter, "StandardPdfPipeline"
        
        elif doc_type == DocumentType.PDF_SCANNED:
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=StandardPdfPipeline,
                        pipeline_options=self.pdf_scanned_options
                    )
                }
            )
            return converter, "StandardPdfPipeline_OCR"
            
        elif doc_type == DocumentType.PDF_MULTIMODAL:
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                        pipeline_options=self.vlm_options
                    )
                }
            )
            return converter, "VlmPipeline"
            
        elif doc_type == DocumentType.IMAGE:
            converter = DocumentConverter(
                format_options={
                    InputFormat.IMAGE: PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                        pipeline_options=self.vlm_options
                    )
                }
            )
            return converter, "VlmPipeline_Image"
            
        elif doc_type == DocumentType.AUDIO_VIDEO:
            # Note: ASR pipeline setup may need different configuration
            converter = DocumentConverter(
                format_options={
                    # This may need adjustment based on actual Docling ASR implementation
                    InputFormat.AUDIO: FormatOption(
                        pipeline_options=self.asr_options
                    )
                }
            )
            return converter, "ASR_Pipeline"
            
        else:  # UNKNOWN or DOCUMENT
            # Use standard pipeline as fallback
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=StandardPdfPipeline,
                        pipeline_options=self.pdf_text_options
                    )
                }
            )
            return converter, "StandardPdfPipeline_Default"

    def _save_to_file(self, 
                    document_result: Any, 
                    file_path: str, 
                    doc_type: DocumentType) -> Optional[str]:
        """
        Save processed document to output file.
        
        Args:
            document_result: Docling conversion result
            file_path: Original file path
            doc_type: Document type
            
        Returns:
            Output file path if successful, None otherwise
        """
        
        if not self.config.save_to_files or not self.output_dir:
            return None
        
        try:
            # Create output filename
            input_file = Path(file_path)
            output_filename = f"{input_file.stem}_processed.{self.config.output_format}"
            output_path = self.output_dir / output_filename
            
            # Export based on format
            if self.config.output_format == "markdown":
                content = document_result.document.export_to_markdown()
            elif self.config.output_format == "json":
                content = document_result.document.export_to_json()
            elif self.config.output_format == "text":
                content = document_result.document.export_to_text()
            else:
                content = document_result.document.export_to_markdown()  # default
                
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                if self.config.output_format == "json":
                    # Parse and pretty-print JSON
                    try:
                        json_data = json.loads(content)
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    except json.JSONDecodeError:
                        # If not valid JSON, write as-is
                        f.write(content)
                else:
                    f.write(content)
            
            logger.info(f"Saved processed document to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save document to file: {e}")
            return None

    def extract_pdf_content(self, file_path: str) -> ExtractionResult:
        """
        Extract content from document using smart Docling processing.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Extraction result with content and metadata
        """
        try:
            # Detect document type for smart pipeline selection
            doc_type = self.detect_document_type(file_path)
            logger.info(f"Detected document type: {doc_type.value} for {file_path}")
            
            # Get appropriate Docling converter
            converter, pipeline_name = self._get_docling_converter(doc_type)
            logger.info(f"Using pipeline: {pipeline_name}")
            
            # Process document with Docling
            result = converter.convert(source=file_path)
            
            # Extract content based on output format
            if self.config.output_format == "markdown":
                content = result.document.export_to_markdown()
            elif self.config.output_format == "json":
                content = result.document.export_to_json()
            else:
                content = result.document.export_to_text()
            
            # Save to file if requested
            output_file_path = self._save_to_file(result, file_path, doc_type)
            
            # Prepare metadata
            metadata = {
                "document_type": doc_type.value,
                "pipeline_used": pipeline_name,
                "file_path": str(file_path),
                "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
                "page_count": getattr(result.document, 'page_count', 1),
                "processing_options": {
                    "ocr_enabled": self.config.enable_ocr,
                    "vlm_enabled": self.config.enable_vlm,
                    "table_extraction": self.config.enable_table_extraction,
                    "image_extraction": self.config.enable_image_extraction,
                    "output_format": self.config.output_format
                },
                "docling_metadata": getattr(result, 'metadata', {})
            }
            
            # Add document-specific metadata if available
            if hasattr(result.document, 'metadata'):
                metadata.update(result.document.metadata)
            
            return ExtractionResult(
                success=True,
                content=content,
                document_type=doc_type,
                pipeline_used=pipeline_name,
                metadata=metadata,
                output_file_path=output_file_path
            )
            
        except Exception as e:
            logger.error(f"Failed to extract content from {file_path}: {e}")
            return ExtractionResult(
                success=False,
                content="",
                document_type=doc_type if 'doc_type' in locals() else DocumentType.UNKNOWN,
                pipeline_used="error",
                metadata={"error": str(e)},
                error_message=str(e)
            )

    def get_supported_extensions(self) -> set:
        """Get set of supported file extensions."""
        return set(self.file_type_mapping.keys())

    def is_supported_file(self, file_path: str) -> bool:
        """Check if file is supported for extraction."""
        extension = Path(file_path).suffix.lower()
        return extension in self.file_type_mapping


def create_pdf_extractor(config: PDFExtractionConfig) -> DoclingExtractor:
    """
    Factory function to create a Docling extractor.
    
    Args:
        config: Extraction configuration
        
    Returns:
        Configured DoclingExtractor instance
    """
    return DoclingExtractor(config)


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Docling document extraction")
    parser.add_argument("file_path", help="Path to document to extract")
    parser.add_argument("--output-dir", default="./extracted_docs", help="Output directory")
    parser.add_argument("--format", choices=["markdown", "json", "text"], 
                        default="markdown", help="Output format")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR")
    parser.add_argument("--no-vlm", action="store_true", help="Disable VLM")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create extraction config
    config = PDFExtractionConfig(
        enable_ocr=not args.no_ocr,
        enable_vlm=not args.no_vlm,
        save_to_files=True,
        output_format=args.format,
        output_directory=args.output_dir
    )
    
    # Create extractor
    extractor = create_pdf_extractor(config)
    
    # Test extraction
    if extractor.is_supported_file(args.file_path):
        print(f"Extracting content from: {args.file_path}")
        
        result = extractor.extract_pdf_content(args.file_path)
        
        if result.success:
            print(f"Extraction successful!")
            print(f"Document type: {result.document_type.value}")
            print(f"Pipeline used: {result.pipeline_used}")
            print(f"Content length: {len(result.content)} characters")
            if result.output_file_path:
                print(f"Saved to: {result.output_file_path}")
        else:
            print(f"Extraction failed: {result.error_message}")
    else:
        print(f"Unsupported file type: {args.file_path}")
        print(f"Supported extensions: {extractor.get_supported_extensions()}")