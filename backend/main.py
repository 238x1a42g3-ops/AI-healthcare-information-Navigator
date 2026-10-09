import os
import sys
import tempfile
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "venv"))

import rag


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
state_lock = Lock()
active_document: str | None = None

app = FastAPI(
    title="AI Healthcare Information Navigator",
    description="Ask questions about an uploaded healthcare PDF.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "document_loaded": active_document is not None,
        "document": active_document,
        "groq_configured": bool(rag.GROQ_API_KEY),
    }


@app.post("/api/documents")
def upload_document(file: UploadFile = File(...)):
    global active_document

    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Please upload a PDF file.")

    temp_path = None
    total_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".pdf",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
            while True:
                block = file.file.read(1024 * 1024)
                if not block:
                    break
                total_bytes += len(block)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="PDF exceeds the 25 MB upload limit.",
                    )
                temp_file.write(block)

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")

        with state_lock:
            if not rag.process_pdf(temp_path):
                raise HTTPException(
                    status_code=422,
                    detail="Could not extract and index this PDF. Check that it contains readable text.",
                )
            active_document = filename
            chunk_count = len(rag.stored_chunks)

        return {
            "document": filename,
            "chunks": chunk_count,
            "message": "PDF is ready for questions.",
        }
    finally:
        file.file.close()
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/api/chat")
def chat(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Enter a question.")
    if not rag.GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="The server is missing GROQ_API_KEY. Add it to the project .env file.",
        )

    with state_lock:
        if active_document is None:
            raise HTTPException(
                status_code=409,
                detail="Upload a healthcare PDF before asking a question.",
            )

        documents = rag.search_chromadb(question)
        context = rag.create_context(documents)
        answer = rag.generate_answer(question, context)

    if answer == "Unable to generate an answer because of a Groq API error.":
        raise HTTPException(
            status_code=502,
            detail="The answer service could not complete the request. Please try again.",
        )

    return {
        "answer": answer,
        "sources": documents,
        "document": active_document,
    }
