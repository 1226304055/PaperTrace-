"""Application service that coordinates ingestion, retrieval, and grounded answers."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

from config import Settings, settings
from image_caption import ImageCaptioner
from pdf_parser import file_sha256, parse_pdf
from storage import MetadataStore, utc_now
from vector_db import PaperVectorDB, SearchHit


SYSTEM_PROMPT = """你是论文证据问答助手。请严格依据提供的检索证据回答。
- 每个关键结论后使用 [1]、[2] 形式标注证据编号。
- 证据不足时直接说明缺少什么信息，不要补写论文中没有的事实。
- 区分论文原文、图表描述和你的归纳。
- 回答优先简洁；用户要求比较时，再按论文或方法组织内容。"""


def compact_excerpt(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    shortened = compact[: limit - 1].rsplit(" ", 1)[0]
    return (shortened or compact[: limit - 1]) + "…"


class RAGService:
    def __init__(
        self,
        store: MetadataStore,
        app_settings: Settings = settings,
        vector_db: Any | None = None,
        llm_client: Any | None = None,
        captioner: ImageCaptioner | None = None,
    ):
        self.store = store
        self.settings = app_settings
        self._vector_db = vector_db
        self._llm_client = llm_client
        self.captioner = captioner or ImageCaptioner(
            store, app_settings, client=llm_client
        )

    @property
    def vector_db(self) -> PaperVectorDB:
        if self._vector_db is None:
            self._vector_db = PaperVectorDB(app_settings=self.settings)
        return self._vector_db

    @property
    def llm_client(self) -> Any:
        if self._llm_client is None:
            from openai import OpenAI

            self._llm_client = OpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                timeout=90,
            )
        return self._llm_client

    def _caption_images(
        self,
        job_id: str,
        filename: str,
        images: list[dict[str, Any]],
        progress_start: int,
        progress_span: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Describe figures concurrently while keeping per-image failure isolation."""
        if not images or not self.captioner.enabled:
            return [], 0

        captions: list[dict[str, Any]] = []
        failures = 0
        total = len(images)
        workers = max(1, min(self.settings.caption_workers, total))

        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="papertrace-caption"
        ) as executor:
            future_to_image = {
                executor.submit(self.captioner.generate_caption, image): image
                for image in images
            }
            for completed, future in enumerate(as_completed(future_to_image), start=1):
                image = future_to_image[future]
                try:
                    caption = future.result()
                except Exception:
                    failures += 1
                else:
                    if caption:
                        metadata = dict(image["metadata"])
                        metadata.update(
                            {
                                "content_type": "image_caption",
                                "source_image_path": image["path"],
                                "image_hash": image["image_hash"],
                            }
                        )
                        captions.append({"text": caption, "metadata": metadata})

                progress = progress_start + int(completed / total * progress_span)
                self.store.update_job(
                    job_id,
                    progress=progress,
                    message=(
                        f"分析 {filename} 的图表 {completed}/{total}"
                        f"（{workers} 路并发）"
                    ),
                )

        captions.sort(
            key=lambda item: (
                int(item["metadata"].get("page", 0)),
                str(item["metadata"].get("source_image_path", "")),
            )
        )
        return captions, failures

    def run_index_job(self, job_id: str, pdf_dir: str | None, force: bool) -> None:
        source_dir = Path(pdf_dir or self.settings.pdf_dir).expanduser().resolve()
        job_started = time.perf_counter()
        parse_ms = 0.0
        caption_ms = 0.0
        vector_ms = 0.0
        try:
            if not source_dir.exists() or not source_dir.is_dir():
                raise ValueError(f"PDF 目录不存在: {source_dir}")
            pdf_files = sorted(source_dir.glob("*.pdf"))
            if not pdf_files:
                raise ValueError(f"目录中没有 PDF 文件: {source_dir}")

            self.store.update_job(
                job_id,
                status="running",
                progress=2,
                message=f"发现 {len(pdf_files)} 篇论文",
            )
            indexed = 0
            skipped = 0
            failed: list[str] = []

            for index, pdf_path in enumerate(pdf_files):
                base_progress = 5 + int(index / len(pdf_files) * 88)
                self.store.update_job(
                    job_id,
                    progress=base_progress,
                    message=f"处理 {pdf_path.name}",
                )
                digest = file_sha256(pdf_path)
                source_path = str(pdf_path.resolve())
                if not force and self.store.document_is_current(source_path, digest):
                    skipped += 1
                    continue

                previous = self.store.get_document_by_source(source_path)
                try:
                    stage_started = time.perf_counter()
                    parsed = parse_pdf(pdf_path)
                    parse_ms += (time.perf_counter() - stage_started) * 1000
                    self.store.upsert_document(
                        doc_id=parsed.doc_id,
                        source_path=parsed.source_path,
                        filename=parsed.filename,
                        file_hash=parsed.file_hash,
                        status="indexing",
                        page_count=parsed.page_count,
                        text_chunks=len(parsed.text_chunks),
                        image_count=len(parsed.images),
                    )

                    per_file_span = max(1, 82 // len(pdf_files))
                    stage_started = time.perf_counter()
                    captions, caption_failures = self._caption_images(
                        job_id,
                        parsed.filename,
                        parsed.images,
                        base_progress,
                        max(1, per_file_span - 8),
                    )
                    caption_ms += (time.perf_counter() - stage_started) * 1000

                    self.store.update_job(
                        job_id,
                        progress=min(94, base_progress + per_file_span - 5),
                        message=f"写入 {parsed.filename} 的向量索引",
                    )

                    stage_started = time.perf_counter()
                    indexed_units = self.vector_db.index_document(
                        parsed.doc_id, parsed.text_chunks, captions
                    )
                    vector_ms += (time.perf_counter() - stage_started) * 1000
                    if indexed_units == 0:
                        raise ValueError("没有提取到可索引的文本或图表描述")
                    if previous and previous["doc_id"] != parsed.doc_id:
                        self.vector_db.delete_document(str(previous["doc_id"]))

                    warning = None
                    if parsed.images and not self.captioner.enabled:
                        warning = "未配置 API Key，已跳过图表语义描述"
                    elif caption_failures:
                        warning = f"{caption_failures} 张图表描述失败，文本索引仍可使用"
                    self.store.upsert_document(
                        doc_id=parsed.doc_id,
                        source_path=parsed.source_path,
                        filename=parsed.filename,
                        file_hash=parsed.file_hash,
                        status="ready",
                        page_count=parsed.page_count,
                        text_chunks=len(parsed.text_chunks),
                        image_count=len(parsed.images),
                        caption_count=len(captions),
                        error=warning,
                        indexed_at=utc_now(),
                    )
                    indexed += 1
                except Exception as exc:
                    failed.append(f"{pdf_path.name}: {exc}")
                    self.store.upsert_document(
                        doc_id=digest[:20],
                        source_path=source_path,
                        filename=pdf_path.name,
                        file_hash=digest,
                        status="failed",
                        error=str(exc),
                    )

            summary = f"新增或更新 {indexed} 篇，跳过 {skipped} 篇"
            status = "completed"
            if failed:
                status = "completed_with_errors"
                summary += f"，失败 {len(failed)} 篇：" + "；".join(failed[:3])
            self.store.update_job(
                job_id, status=status, progress=100, message=summary
            )
            self.store.add_trace(
                operation="index",
                status="success" if status == "completed" else "partial",
                duration_ms=(time.perf_counter() - job_started) * 1000,
                metadata={
                    "job_id": job_id,
                    "pdf_count": len(pdf_files),
                    "indexed": indexed,
                    "skipped": skipped,
                    "failed": len(failed),
                    "parse_ms": round(parse_ms, 2),
                    "caption_ms": round(caption_ms, 2),
                    "vector_ms": round(vector_ms, 2),
                    "caption_workers": self.settings.caption_workers,
                },
                error="；".join(failed[:3]) if failed else None,
            )
        except Exception as exc:
            self.store.update_job(
                job_id, status="failed", progress=100, message=str(exc)
            )
            self.store.add_trace(
                operation="index",
                status="error",
                duration_ms=(time.perf_counter() - job_started) * 1000,
                metadata={
                    "job_id": job_id,
                    "parse_ms": round(parse_ms, 2),
                    "caption_ms": round(caption_ms, 2),
                    "vector_ms": round(vector_ms, 2),
                },
                error=str(exc),
            )

    def _citations(self, hits: list[SearchHit]) -> list[dict[str, Any]]:
        citations = []
        for rank, hit in enumerate(hits, start=1):
            image_url = None
            if hit.image_path:
                try:
                    relative = Path(hit.image_path).resolve().relative_to(
                        self.settings.images_dir.resolve()
                    )
                    image_url = "/artifacts/" + relative.as_posix()
                except ValueError:
                    image_url = None
            citations.append(
                {
                    "rank": rank,
                    "source": hit.source,
                    "page": hit.page,
                    "snippet": compact_excerpt(hit.text, 360),
                    "content_type": hit.content_type,
                    "score": hit.score,
                    "image_url": image_url,
                }
            )
        return citations

    def _context(self, hits: list[SearchHit]) -> str:
        blocks = []
        length = 0
        for rank, hit in enumerate(hits, start=1):
            label = "图表描述" if hit.content_type == "image_caption" else "原文"
            block = f"[{rank}] {hit.source} 第 {hit.page} 页（{label}）\n{hit.text}"
            if blocks and length + len(block) > self.settings.max_context_chars:
                break
            blocks.append(block)
            length += len(block)
        return "\n\n".join(blocks)

    def _messages(
        self, query: str, session_id: str, context: str
    ) -> list[dict[str, str]]:
        history = self.store.get_messages(
            session_id, limit=self.settings.history_turns * 2
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in history:
            if item["role"] in {"user", "assistant"}:
                messages.append({"role": item["role"], "content": item["content"]})
        messages.append(
            {
                "role": "user",
                "content": f"检索证据：\n{context}\n\n问题：{query}",
            }
        )
        return messages

    def prepare_answer(
        self,
        query: str,
        session_id: str | None,
        k: int,
        doc_ids: list[str] | None,
    ) -> tuple[str, list[SearchHit], list[dict[str, Any]], list[dict[str, str]]]:
        session_id = self.store.ensure_session(session_id)
        has_documents = any(
            item["status"] == "ready" for item in self.store.list_documents()
        )
        hits = self.vector_db.retrieve(query, k=k, doc_ids=doc_ids) if has_documents else []
        citations = self._citations(hits)
        messages = self._messages(query, session_id, self._context(hits))
        self.store.add_message(session_id, "user", query)
        return session_id, hits, citations, messages

    def _offline_answer(self, hits: list[SearchHit]) -> str:
        if not hits:
            return "没有检索到相关证据。请先索引论文，或换一个更具体的问题。"
        summaries = [
            f"[{index}] {compact_excerpt(hit.text, 220)}"
            for index, hit in enumerate(hits[:3], start=1)
        ]
        return (
            "当前未配置大模型 API Key，因此先返回相关证据摘要：\n\n"
            + "\n\n".join(summaries)
        )

    def answer(
        self,
        query: str,
        session_id: str | None = None,
        k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        retrieval_started = time.perf_counter()
        resolved_session = session_id
        citations: list[dict[str, Any]] = []
        retrieval_ms = 0.0
        generation_ms = 0.0
        input_tokens: int | None = None
        output_tokens: int | None = None
        total_tokens: int | None = None
        try:
            resolved_session, hits, citations, messages = self.prepare_answer(
                query, session_id, k or self.settings.top_k, doc_ids
            )
            retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
            generation_started = time.perf_counter()
            if not self.settings.api_key:
                answer = self._offline_answer(hits)
            else:
                error: Exception | None = None
                answer = ""
                for attempt in range(self.settings.retry_attempts):
                    try:
                        response = self.llm_client.chat.completions.create(
                            model=self.settings.chat_model,
                            messages=messages,
                            max_tokens=1200,
                            temperature=0.2,
                        )
                        answer = response.choices[0].message.content.strip()
                        usage = getattr(response, "usage", None)
                        if usage:
                            input_tokens = getattr(usage, "prompt_tokens", None)
                            output_tokens = getattr(usage, "completion_tokens", None)
                            total_tokens = getattr(usage, "total_tokens", None)
                        break
                    except Exception as exc:
                        error = exc
                        if attempt + 1 < self.settings.retry_attempts:
                            time.sleep(
                                self.settings.retry_backoff_seconds * (2**attempt)
                            )
                if not answer:
                    raise RuntimeError(f"回答生成失败: {error}") from error
            generation_ms = (time.perf_counter() - generation_started) * 1000
            self.store.add_message(resolved_session, "assistant", answer, citations)
            self.store.add_trace(
                operation="query",
                session_id=resolved_session,
                status="success",
                duration_ms=(time.perf_counter() - started) * 1000,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                citation_count=len(citations),
                metadata={
                    "stream": False,
                    "model": self.settings.chat_model,
                    "k": k or self.settings.top_k,
                    "usage_source": "provider" if total_tokens is not None else "none",
                },
            )
            return {
                "response": answer,
                "session_id": resolved_session,
                "citations": citations,
            }
        except Exception as exc:
            self.store.add_trace(
                operation="query",
                session_id=resolved_session,
                status="error",
                duration_ms=(time.perf_counter() - started) * 1000,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms or None,
                citation_count=len(citations),
                metadata={"stream": False, "model": self.settings.chat_model},
                error=str(exc),
            )
            raise

    @staticmethod
    def _sse(event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def stream_answer(
        self,
        query: str,
        session_id: str | None = None,
        k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> Iterator[str]:
        started = time.perf_counter()
        retrieval_started = time.perf_counter()
        resolved_session = session_id
        citations: list[dict[str, Any]] = []
        retrieval_ms = 0.0
        generation_ms = 0.0
        try:
            resolved_session, hits, citations, messages = self.prepare_answer(
                query, session_id, k or self.settings.top_k, doc_ids
            )
            retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
            yield self._sse(
                "meta", {"session_id": resolved_session, "citations": citations}
            )
            generation_started = time.perf_counter()
            if not self.settings.api_key:
                answer = self._offline_answer(hits)
                yield self._sse("chunk", {"text": answer})
            else:
                answer_parts: list[str] = []
                error: Exception | None = None
                for attempt in range(self.settings.retry_attempts):
                    emitted = False
                    try:
                        stream = self.llm_client.chat.completions.create(
                            model=self.settings.chat_model,
                            messages=messages,
                            max_tokens=1200,
                            temperature=0.2,
                            stream=True,
                        )
                        for chunk in stream:
                            text = chunk.choices[0].delta.content or ""
                            if text:
                                emitted = True
                                answer_parts.append(text)
                                yield self._sse("chunk", {"text": text})
                        error = None
                        break
                    except Exception as exc:
                        error = exc
                        if emitted or attempt + 1 >= self.settings.retry_attempts:
                            break
                        time.sleep(self.settings.retry_backoff_seconds * (2**attempt))
                if error:
                    raise RuntimeError(f"流式回答失败: {error}") from error
                answer = "".join(answer_parts).strip()
            generation_ms = (time.perf_counter() - generation_started) * 1000
            self.store.add_message(resolved_session, "assistant", answer, citations)
            estimated_input = max(
                1, sum(len(item["content"]) for item in messages) // 3
            )
            estimated_output = max(1, len(answer) // 3)
            self.store.add_trace(
                operation="query",
                session_id=resolved_session,
                status="success",
                duration_ms=(time.perf_counter() - started) * 1000,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                input_tokens=estimated_input,
                output_tokens=estimated_output,
                total_tokens=estimated_input + estimated_output,
                citation_count=len(citations),
                metadata={
                    "stream": True,
                    "model": self.settings.chat_model,
                    "k": k or self.settings.top_k,
                    "usage_source": "estimated_from_characters",
                },
            )
            yield self._sse("done", {"session_id": resolved_session})
        except Exception as exc:
            self.store.add_trace(
                operation="query",
                session_id=resolved_session,
                status="error",
                duration_ms=(time.perf_counter() - started) * 1000,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms or None,
                citation_count=len(citations),
                metadata={"stream": True, "model": self.settings.chat_model},
                error=str(exc),
            )
            yield self._sse("error", {"message": str(exc)})
