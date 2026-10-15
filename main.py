"""FastAPI entry point for PaperTrace."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hashlib
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app_models import (
    HealthResponse,
    IndexAccepted,
    IndexRequest,
    JobResponse,
    MessageResponse,
    QueryRequest,
    QueryResponse,
)
from config import settings
from rag_service import RAGService
from storage import MetadataStore


settings.ensure_directories()
store = MetadataStore(settings.sqlite_db_path)
service = RAGService(store, settings)
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="papertrace-index")


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="PaperTrace",
    version="2.0.0",
    description="面向实验室论文库的图文证据问答与对比工具",
    lifespan=lifespan,
)
app.mount(
    "/artifacts",
    StaticFiles(directory=str(settings.images_dir), check_dir=False),
    name="artifacts",
)


@app.get("/", include_in_schema=False)
def web_app() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    documents = store.list_documents()
    ready = [item for item in documents if item["status"] == "ready"]
    return HealthResponse(
        status="ok",
        llm_configured=bool(settings.api_key),
        indexed_documents=len(ready),
        indexed_units=sum(
            int(item["text_chunks"]) + int(item["caption_count"]) for item in ready
        ),
        models={
            "chat": settings.chat_model,
            "vision": settings.vision_model,
            "embedding": settings.embedding_model,
        },
    )


@app.post("/index", response_model=IndexAccepted, status_code=202)
@app.post("/api/index", response_model=IndexAccepted, status_code=202)
def index_papers(request: IndexRequest) -> IndexAccepted:
    source_dir = str(Path(request.pdf_dir or settings.pdf_dir).expanduser().resolve())
    job_id = store.create_job(source_dir, request.force)
    executor.submit(service.run_index_job, job_id, source_dir, request.force)
    return IndexAccepted(job_id=job_id)


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def job_status(job_id: str) -> JobResponse:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    return JobResponse(**job)


@app.get("/api/documents")
def documents() -> list[dict]:
    return store.list_documents()


@app.get("/api/observability/summary")
def observability_summary() -> dict:
    return store.trace_summary()


@app.get("/api/observability/traces")
def observability_traces(limit: int = 50) -> list[dict]:
    return store.list_traces(limit)


@app.post("/api/documents/upload", response_model=IndexAccepted, status_code=202)
async def upload_documents(files: list[UploadFile] = File(...)) -> IndexAccepted:
    if not files:
        raise HTTPException(status_code=400, detail="请选择至少一个 PDF 文件")
    max_bytes = settings.max_upload_mb * 1024 * 1024
    saved = 0
    for upload in files:
        original_name = Path(upload.filename or "paper.pdf").name
        if Path(original_name).suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail=f"仅支持 PDF: {original_name}")
        content = await upload.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{original_name} 超过 {settings.max_upload_mb} MB",
            )
        if not content.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail=f"文件格式无效: {original_name}")
        destination = settings.pdf_dir / original_name
        if destination.exists() and destination.read_bytes() != content:
            digest = hashlib.sha256(content).hexdigest()[:8]
            destination = settings.pdf_dir / f"{destination.stem}-{digest}.pdf"
        if not destination.exists():
            destination.write_bytes(content)
            saved += 1
    job_id = store.create_job(str(settings.pdf_dir), force=False)
    executor.submit(service.run_index_job, job_id, str(settings.pdf_dir), False)
    return IndexAccepted(job_id=job_id, status=f"queued:{saved}")


@app.post("/ask", response_model=QueryResponse)
@app.post("/api/ask", response_model=QueryResponse)
def ask(request: QueryRequest) -> QueryResponse:
    try:
        result = service.answer(
            request.message, request.session_id, request.k, request.doc_ids
        )
        return QueryResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/ask/stream")
def ask_stream(request: QueryRequest) -> StreamingResponse:
    return StreamingResponse(
        service.stream_answer(
            request.message, request.session_id, request.k, request.doc_ids
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get(
    "/api/sessions/{session_id}/messages", response_model=list[MessageResponse]
)
def session_messages(session_id: str) -> list[MessageResponse]:
    return [MessageResponse(**item) for item in store.get_messages(session_id, 100)]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
