import os
import shutil
from tempfile import NamedTemporaryFile
from fastapi import APIRouter, File, HTTPException, UploadFile
from app.core.dependencies import ingest_pdf_use_case, answer_query_use_case
from app.infrastructure.web.api_models import AskRequest, AskResponse

router = APIRouter()

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    temp_file_path = None

    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        ingest_pdf_use_case.execute(temp_file_path)

        return {"message": "PDF ingested successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    try:
        result = answer_query_use_case.execute(
            question=request.question,
            top_k=request.top_k,
            history=request.history
        )
        
        return AskResponse(
            question=result.question,
            answer=result.answer,
            context=result.context,
            history=result.history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))