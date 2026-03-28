import json
import os
import shutil
from tempfile import NamedTemporaryFile
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from app.core.dependencies import ingest_pdf_use_case, answer_query_use_case
from app.infrastructure.web.api_models import AskRequest, AskResponse

router = APIRouter()

@router.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    """Endpoint to upload a PDF file for ingestion."""
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
    """Endpoint to ask a question based on ingested PDFs."""
    try:
        result = answer_query_use_case.answer(
            question=request.question,
            top_k=request.top_k,
            history=request.history
        )
        
        updated_history = (request.history or []) + [
            {"role": "user", "content": result.question},
            {"role": "assistant", "content": result.answer},
        ]
        
        return AskResponse(
            question=result.question,
            answer=result.answer,
            context=result.context,
            history=updated_history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/ask/stream")
async def ask_question_stream(request: AskRequest):
    """Endpoint to ask a question and yield a streaming response."""
    def generate():
        full_answer = ""
        try:
            print("Preparing answer...")
            context, rewritten_question = answer_query_use_case.prepare_answer(
                question=request.question,
                top_k=request.top_k,
                history=request.history
            )
            
            print(f"Rewritten question: {rewritten_question}, Context chunks: {len(context)}")
            
            for chunk in answer_query_use_case.stream_answer(
                question=request.question,
                context=context,
                top_k=request.top_k,
                history=request.history
            ):
                full_answer += chunk

                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return

        updated_history = (request.history or []) + [
            {"role": "user", "content": rewritten_question},
            {"role": "assistant", "content": full_answer},
        ]

        final_payload = AskResponse(
            question=request.question,
            answer=full_answer,
            context=context,
            history=updated_history
        ).model_dump()

        yield f"data: {json.dumps({'type': 'done', 'payload': final_payload})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")