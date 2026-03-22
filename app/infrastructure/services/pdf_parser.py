from docling.document_converter import DocumentConverter

class DoclingParser:
    
    def __init__(self):
        self.converter = DocumentConverter()
        
    def parse(self, document_path: str) -> str:
        """Parse a document and return its content as Document."""
        result = self.converter.convert(document_path)
        
        if not result or not result.document:
            raise ValueError("Failed to parse document")
    
        markdown = result.document.export_to_markdown()
        
        if not markdown:
            raise ValueError("Parsed document is empty")
        
        return markdown
        