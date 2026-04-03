import os
from tempfile import NamedTemporaryFile

from app.core.dependencies import get_ingest_pdf_use_case


def ingest_pdf_job(file_bytes: bytes, filename: str) -> dict:
    temp_file_path = None

    try:
        suffix = os.path.splitext(filename or "upload.pdf")[1] or ".pdf"

        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        get_ingest_pdf_use_case().execute(temp_file_path)

        return {
            "filename": filename,
            "message": "PDF ingested successfully.",
        }
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
