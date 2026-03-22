from fastapi import FastAPI
from app.infrastructure.web.routes import router

app = FastAPI(title="RAG Engine Backend")
app.include_router(router=router)