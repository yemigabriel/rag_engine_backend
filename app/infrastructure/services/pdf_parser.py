from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.document_converter import DocumentConverter, PdfFormatOption


class DoclingParser:
    def __init__(self, converter=None, ocr_converter=None):
        self.converter = converter or self._build_converter(enable_ocr=False)
        self._ocr_converter = ocr_converter

    def parse(self, document_path: str) -> str:
        """Parse a document and fall back to OCR when plain extraction is empty."""
        markdown = self._extract_markdown(self.converter, document_path)
        if markdown:
            return markdown

        ocr_converter = self._get_ocr_converter()
        markdown = self._extract_markdown(ocr_converter, document_path)
        if markdown:
            return markdown

        raise ValueError("Parsed document is empty")

    def _get_ocr_converter(self):
        if self._ocr_converter is None:
            self._ensure_ocr_dependencies()
            self._ocr_converter = self._build_converter(enable_ocr=True)
        return self._ocr_converter

    @staticmethod
    def _build_converter(enable_ocr: bool) -> DocumentConverter:
        pipeline_options = PdfPipelineOptions(
            generate_parsed_pages=False,
            do_formula_enrichment=False,
            do_code_enrichment=False,
            do_picture_classification=False,
            ocr_batch_size=2,
            layout_batch_size=2,
            table_batch_size=2,
            queue_max_size=20,
        )
        pipeline_options.do_ocr = enable_ocr
        if enable_ocr:
            pipeline_options.ocr_options = RapidOcrOptions()

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend,
                )
            }
        )

    @staticmethod
    def _ensure_ocr_dependencies() -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR  # noqa: F401
        except Exception as exc:
            raise RuntimeError(f"RapidOCR import failed: {exc}") from exc

        try:
            import cv2  # noqa: F401
        except Exception as exc:
            raise RuntimeError(f"OpenCV import failed: {exc}") from exc

    @staticmethod
    def _extract_markdown(converter: DocumentConverter, document_path: str) -> str:
        result = None
        try:
            result = converter.convert(document_path)
            if not result or not result.document:
                return ""

            markdown = result.document.export_to_markdown()
            return markdown.strip() if markdown else ""
        finally:
            backend = getattr(getattr(result, "input", None), "_backend", None)
            if backend and hasattr(backend, "unload"):
                backend.unload()
