"""API request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from config import settings


class IndexRequest(BaseModel):
    pdf_dir: str | None = None
    force: bool = False


class IndexAccepted(BaseModel):
    job_id: str
    status: str = "queued"


class QueryRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    k: int = Field(default=settings.top_k, ge=1, le=12)
    doc_ids: list[str] | None = None


class Citation(BaseModel):
    rank: int
    source: str
    page: int
    snippet: str
    content_type: str
    score: float
    image_url: str | None = None


class QueryResponse(BaseModel):
    response: str
    session_id: str
    citations: list[Citation]


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    indexed_documents: int
    indexed_units: int
    models: dict[str, str]


class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    source_dir: str
    force: int
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    role: str
    content: str
    citations: list[dict[str, Any]]
    created_at: str
