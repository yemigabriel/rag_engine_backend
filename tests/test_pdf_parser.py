from types import SimpleNamespace

import pytest

from app.infrastructure.services.pdf_parser import DoclingParser


class FakeDocument:
    def __init__(self, markdown: str):
        self.markdown = markdown

    def export_to_markdown(self):
        return self.markdown


class FakeConverter:
    def __init__(self, markdown: str):
        self.markdown = markdown
        self.calls = []

    def convert(self, document_path: str):
        self.calls.append(document_path)
        return SimpleNamespace(document=FakeDocument(self.markdown))


def test_docling_parser_returns_non_ocr_result_when_text_exists():
    standard_converter = FakeConverter("Digital PDF text")
    ocr_converter = FakeConverter("OCR text")
    parser = DoclingParser(converter=standard_converter, ocr_converter=ocr_converter)

    result = parser.parse("/tmp/sample.pdf")

    assert result == "Digital PDF text"
    assert standard_converter.calls == ["/tmp/sample.pdf"]
    assert ocr_converter.calls == []


def test_docling_parser_falls_back_to_ocr_when_initial_parse_is_empty():
    standard_converter = FakeConverter("   ")
    ocr_converter = FakeConverter("Scanned PDF text")
    parser = DoclingParser(converter=standard_converter, ocr_converter=ocr_converter)

    result = parser.parse("/tmp/sample.pdf")

    assert result == "Scanned PDF text"
    assert standard_converter.calls == ["/tmp/sample.pdf"]
    assert ocr_converter.calls == ["/tmp/sample.pdf"]


def test_docling_parser_raises_when_both_paths_return_empty():
    parser = DoclingParser(
        converter=FakeConverter(""),
        ocr_converter=FakeConverter(""),
    )

    with pytest.raises(ValueError, match="Parsed document is empty"):
        parser.parse("/tmp/sample.pdf")
