from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import RapidOcrOptions, PdfPipelineOptions
import onnxruntime

class DoclingParser:
    
    def __init__(self):
        try:
            from rapidocr_onnxruntime import RapidOCR
        except Exception as e:
            raise RuntimeError(f"RapidOCR import failed: {e}")

        try:
            import cv2
        except Exception as e:
            raise RuntimeError(f"OpenCV import failed: {e}")


        self.converter = DocumentConverter()
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = RapidOcrOptions()

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )
        
        
    def parse(self, document_path: str) -> str:
        """Parse a document and return its content as Document."""
        result = self.converter.convert(document_path)
        
        if not result or not result.document:
            raise ValueError("Failed to parse document")
    
        markdown = result.document.export_to_markdown()
        
        if not markdown:
            raise ValueError("Parsed document is empty")
        
        return markdown
        