"""SQLite persistence for documents, jobs, caches, and chat sessions."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MetadataStore:
    """Small thread-safe persistence layer used by the local FastAPI process."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL UNIQUE,
            filename TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            page_count INTEGER NOT NULL DEFAULT 0,
            text_chunks INTEGER NOT NULL DEFAULT 0,
            image_count INTEGER NOT NULL DEFAULT 0,
            caption_count INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            indexed_at TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);

        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            progress INTEGER NOT NULL DEFAULT 0,
            message TEXT NOT NULL DEFAULT '',
            source_dir TEXT NOT NULL,
            force INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            citations_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

        CREATE TABLE IF NOT EXISTS cache (
            cache_key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL UNIQUE,
            operation TEXT NOT NULL,
            session_id TEXT,
            status TEXT NOT NULL,
            duration_ms REAL NOT NULL,
            retrieval_ms REAL,
            generation_ms REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            citation_count INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            error TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_traces_operation ON traces(operation, created_at DESC);
        """
        with self._lock, self._connection() as connection:
            connection.executescript(schema)
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(documents)").fetchall()
            }
            if "caption_count" not in columns:
                connection.execute(
                    "ALTER TABLE documents ADD COLUMN caption_count INTEGER NOT NULL DEFAULT 0"
                )

    def create_job(self, source_dir: str, force: bool = False) -> str:
        job_id = uuid.uuid4().hex
        now = utc_now()
        with self._lock, self._connection() as connection:
            connection.execute(
                """INSERT INTO jobs
                   (job_id, status, progress, message, source_dir, force, created_at, updated_at)
                   VALUES (?, 'queued', 0, '等待处理', ?, ?, ?, ?)""",
                (job_id, source_dir, int(force), now, now),
            )
        return job_id

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: int | None = None,
        message: str | None = None,
    ) -> None:
        fields: list[str] = []
        values: list[Any] = []
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if progress is not None:
            fields.append("progress = ?")
            values.append(max(0, min(100, progress)))
        if message is not None:
            fields.append("message = ?")
            values.append(message)
        fields.append("updated_at = ?")
        values.append(utc_now())
        values.append(job_id)
        with self._lock, self._connection() as connection:
            connection.execute(
                f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?", values
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return dict(row) if row else None

    def upsert_document(
        self,
        *,
        doc_id: str,
        source_path: str,
        filename: str,
        file_hash: str,
        status: str,
        page_count: int = 0,
        text_chunks: int = 0,
        image_count: int = 0,
        caption_count: int = 0,
        error: str | None = None,
        indexed_at: str | None = None,
    ) -> None:
        now = utc_now()
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO documents
                    (doc_id, source_path, filename, file_hash, status, page_count,
                     text_chunks, image_count, caption_count, error, indexed_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_path) DO UPDATE SET
                    doc_id = excluded.doc_id,
                    filename = excluded.filename,
                    file_hash = excluded.file_hash,
                    status = excluded.status,
                    page_count = excluded.page_count,
                    text_chunks = excluded.text_chunks,
                    image_count = excluded.image_count,
                    caption_count = excluded.caption_count,
                    error = excluded.error,
                    indexed_at = excluded.indexed_at,
                    updated_at = excluded.updated_at
                """,
                (
                    doc_id,
                    source_path,
                    filename,
                    file_hash,
                    status,
                    page_count,
                    text_chunks,
                    image_count,
                    caption_count,
                    error,
                    indexed_at,
                    now,
                ),
            )

    def document_is_current(self, source_path: str, file_hash: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                """SELECT 1 FROM documents
                   WHERE source_path = ? AND file_hash = ? AND status = 'ready'""",
                (source_path, file_hash),
            ).fetchone()
        return row is not None

    def get_document_by_source(self, source_path: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE source_path = ?", (source_path,)
            ).fetchone()
        return dict(row) if row else None

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def ensure_session(self, session_id: str | None = None) -> str:
        session_id = session_id or uuid.uuid4().hex
        now = utc_now()
        with self._lock, self._connection() as connection:
            connection.execute(
                """INSERT INTO sessions(session_id, created_at, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at""",
                (session_id, now, now),
            )
        return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Iterable[dict[str, Any]] | None = None,
    ) -> None:
        self.ensure_session(session_id)
        now = utc_now()
        with self._lock, self._connection() as connection:
            connection.execute(
                """INSERT INTO messages
                   (session_id, role, content, citations_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    session_id,
                    role,
                    content,
                    json.dumps(list(citations or []), ensure_ascii=False),
                    now,
                ),
            )

    def get_messages(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT role, content, citations_json, created_at
                   FROM messages WHERE session_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        messages = []
        for row in reversed(rows):
            item = dict(row)
            item["citations"] = json.loads(item.pop("citations_json"))
            messages.append(item)
        return messages

    def cache_get(self, key: str) -> str | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM cache WHERE cache_key = ?", (key,)
            ).fetchone()
        return str(row["value"]) if row else None

    def cache_set(self, key: str, value: str) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                """INSERT INTO cache(cache_key, value, created_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(cache_key) DO UPDATE SET
                       value = excluded.value,
                       created_at = excluded.created_at""",
                (key, value, utc_now()),
            )

    def add_trace(
        self,
        *,
        operation: str,
        status: str,
        duration_ms: float,
        session_id: str | None = None,
        retrieval_ms: float | None = None,
        generation_ms: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        citation_count: int = 0,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> str:
        trace_id = uuid.uuid4().hex
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO traces (
                    trace_id, operation, session_id, status, duration_ms,
                    retrieval_ms, generation_ms, input_tokens, output_tokens,
                    total_tokens, citation_count, metadata_json, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    operation,
                    session_id,
                    status,
                    round(duration_ms, 2),
                    round(retrieval_ms, 2) if retrieval_ms is not None else None,
                    round(generation_ms, 2) if generation_ms is not None else None,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    citation_count,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    error,
                    utc_now(),
                ),
            )
        return trace_id

    def list_traces(self, limit: int = 50) -> list[dict[str, Any]]:
        safe_limit = max(1, min(500, limit))
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT trace_id, operation, session_id, status, duration_ms,
                          retrieval_ms, generation_ms, input_tokens, output_tokens,
                          total_tokens, citation_count, metadata_json, error, created_at
                   FROM traces ORDER BY id DESC LIMIT ?""",
                (safe_limit,),
            ).fetchall()
        traces = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            traces.append(item)
        return traces

    def trace_summary(self, limit: int = 200) -> dict[str, Any]:
        traces = self.list_traces(limit)
        query_traces = [item for item in traces if item["operation"] == "query"]
        index_traces = [item for item in traces if item["operation"] == "index"]
        successful_queries = [
            item for item in query_traces if item["status"] == "success"
        ]
        durations = sorted(float(item["duration_ms"]) for item in successful_queries)

        def percentile(proportion: float) -> float:
            if not durations:
                return 0.0
            index = min(len(durations) - 1, round((len(durations) - 1) * proportion))
            return round(durations[index], 2)

        return {
            "trace_count": len(traces),
            "query_count": len(query_traces),
            "index_count": len(index_traces),
            "query_success_rate": round(
                len(successful_queries) / len(query_traces), 4
            ) if query_traces else 0.0,
            "p50_query_latency_ms": percentile(0.5),
            "p95_query_latency_ms": percentile(0.95),
            "total_tokens": sum(int(item["total_tokens"] or 0) for item in traces),
            "average_citations": round(
                sum(int(item["citation_count"]) for item in successful_queries)
                / len(successful_queries),
                2,
            ) if successful_queries else 0.0,
        }
