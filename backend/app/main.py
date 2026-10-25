from zipfile import BadZipFile

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import MAX_UPLOAD_BYTES
from app.rag.generator import answer_question
from app.rag.ingestion import read_zip_documents
from app.rag.vector_store import rebuild_index
from app.schemas import (
    ChatRequest,
    ChatResponse,
    IngestResponse,
)

app = FastAPI(
    title="DevDocs Copilot API",
    version="1.0.0",
    description="RAG-powered developer documentation chatbot",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a ZIP file.",
        )

    zip_bytes = await file.read()

    if len(zip_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="ZIP file exceeds the 50 MB upload limit.",
        )

    try:
        chunks = read_zip_documents(zip_bytes)
    except BadZipFile:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid ZIP archive.",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=413,
            detail=str(exc),
        )

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No supported, readable source files were found.",
        )

    try:
        count = rebuild_index(chunks)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {type(exc).__name__}",
        )

    return IngestResponse(
        message="Repository indexed successfully.",
        chunks_indexed=count,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = answer_question(
            question=request.question,
            top_k=request.top_k,
        )
        return ChatResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chat request failed: {type(exc).__name__}",
        )