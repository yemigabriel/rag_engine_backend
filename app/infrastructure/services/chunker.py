from typing import List
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.domain.entities import Chunk

class LangChainChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            separators=["\n\n", "\n", " ", ""]
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, text: str) -> List[Chunk]:
        document_id = str(uuid.uuid4())
        split_texts = self.splitter.split_text(text)
        
        chunks = [
            Chunk(
                id=str(uuid.uuid4()), 
                text=split_text, 
                source_document_id=document_id
            ) 
            for split_text in split_texts
        ]
        
        return chunks