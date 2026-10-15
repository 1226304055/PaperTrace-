"""Chroma-backed hybrid retrieval with deterministic document replacement."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from config import Settings, settings


@dataclass
class SearchHit:
    text: str
    source: str
    page: int
    content_type: str
    doc_id: str
    score: float
    image_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tokenize(text: str) -> set[str]:
    """Tokenize English words and Chinese characters for a lightweight lexical signal."""
    return set(re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower()))


def lexical_overlap(query: str, text: str) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    return len(query_tokens & tokenize(text)) / len(query_tokens)


def idf_lexical_scores(query: str, texts: list[str]) -> list[float]:
    """Return IDF-weighted query-token coverage for every candidate text."""
    query_tokens = tokenize(query)
    if not query_tokens or not texts:
        return [0.0 for _ in texts]
    tokenized = [tokenize(text) for text in texts]
    document_count = len(tokenized)
    weights = {}
    for token in query_tokens:
        document_frequency = sum(token in tokens for tokens in tokenized)
        rarity = math.log(
            (document_count + 1) / (document_frequency + 0.5)
        ) + 1.0
        # Longer Latin terms and numeric identifiers usually carry more intent
        # than a single Chinese character (for example ConvNeXt-Tiny or 0.8mm).
        specificity = min(4.0, max(1.0, float(len(token))))
        weights[token] = rarity * specificity
    denominator = sum(weights.values()) or 1.0
    return [
        sum(weights[token] for token in query_tokens & tokens) / denominator
        for tokens in tokenized
    ]


def rerank_hits(query: str, hits: Iterable[SearchHit], limit: int) -> list[SearchHit]:
    reranked = []
    for hit in hits:
        semantic = max(0.0, min(1.0, hit.score))
        lexical = lexical_overlap(query, hit.text)
        hit.score = round(semantic * 0.8 + lexical * 0.2, 4)
        reranked.append(hit)
    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked[:limit]


class FastEmbedAdapter:
    """Minimal LangChain embedding interface backed directly by FastEmbed."""

    def __init__(self, model_name: str, cache_dir: str | Path):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir),
            lazy_load=True,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(self.model.embed([text])).tolist()


class PaperVectorDB:
    """Lazy Chroma wrapper so the web app can start before a model is downloaded."""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        app_settings: Settings = settings,
        embedding: Any | None = None,
    ):
        self.settings = app_settings
        self.persist_dir = str(persist_dir or app_settings.chroma_db_dir)
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        if embedding is None:
            embedding = FastEmbedAdapter(
                app_settings.embedding_model, app_settings.embedding_cache_dir
            )
        from langchain_chroma import Chroma

        self.db = Chroma(
            collection_name="papertrace",
            persist_directory=self.persist_dir,
            embedding_function=embedding,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return int(self.db._collection.count())

    def delete_document(self, doc_id: str) -> None:
        self.db.delete(where={"doc_id": doc_id})

    def index_document(
        self,
        doc_id: str,
        text_chunks: list[dict[str, Any]],
        image_captions: list[dict[str, Any]],
    ) -> int:
        from langchain_core.documents import Document

        self.delete_document(doc_id)
        items = [*text_chunks, *image_captions]
        if not items:
            return 0
        documents = []
        ids = []
        for index, item in enumerate(items):
            metadata = {
                key: value if isinstance(value, (str, int, float, bool)) else str(value)
                for key, value in item.get("metadata", {}).items()
            }
            documents.append(Document(page_content=item["text"], metadata=metadata))
            ids.append(f"{doc_id}:{metadata.get('content_type', 'text')}:{index}")
        self.db.add_documents(documents=documents, ids=ids)
        return len(documents)

    def retrieve(
        self,
        query: str,
        *,
        k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        k = k or self.settings.top_k
        candidate_count = max(k, self.settings.retrieval_candidates)
        chroma_filter = None
        if doc_ids:
            chroma_filter = (
                {"doc_id": doc_ids[0]}
                if len(doc_ids) == 1
                else {"doc_id": {"$in": doc_ids}}
            )
        results = self.db.similarity_search_with_score(
            query,
            k=candidate_count,
            filter=chroma_filter,
        )
        hits: list[SearchHit] = []
        for document, distance in results:
            metadata = document.metadata
            distance_value = float(distance)
            semantic_score = 1.0 / (1.0 + max(0.0, distance_value))
            if math.isnan(semantic_score):
                semantic_score = 0.0
            hits.append(
                SearchHit(
                    text=document.page_content,
                    source=str(metadata.get("source", "unknown")),
                    page=int(metadata.get("page", 0)),
                    content_type=str(metadata.get("content_type", "text")),
                    doc_id=str(metadata.get("doc_id", "")),
                    score=semantic_score,
                    image_path=metadata.get("source_image_path"),
                )
            )

        lexical_data = self.db.get(
            where=chroma_filter,
            limit=self.settings.lexical_scan_limit,
            include=["documents", "metadatas"],
        )
        lexical_documents = lexical_data.get("documents") or []
        lexical_metadatas = lexical_data.get("metadatas") or []
        lexical_scores = idf_lexical_scores(query, lexical_documents)

        def hit_key(hit: SearchHit) -> tuple[str, int, str, str]:
            return (hit.doc_id, hit.page, hit.content_type, hit.text)

        merged = {hit_key(hit): hit for hit in hits}
        for text, metadata, lexical_score in zip(
            lexical_documents, lexical_metadatas, lexical_scores
        ):
            if lexical_score <= 0:
                continue
            candidate = SearchHit(
                text=str(text),
                source=str(metadata.get("source", "unknown")),
                page=int(metadata.get("page", 0)),
                content_type=str(metadata.get("content_type", "text")),
                doc_id=str(metadata.get("doc_id", "")),
                score=0.65 * lexical_score,
                image_path=metadata.get("source_image_path"),
            )
            key = hit_key(candidate)
            if key in merged:
                semantic_score = merged[key].score
                merged[key].score = 0.65 * semantic_score + 0.35 * lexical_score
            else:
                merged[key] = candidate

        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        for hit in ranked:
            hit.score = round(max(0.0, min(1.0, hit.score)), 4)
        return ranked[:k]


def build_index(
    texts: list[dict[str, Any]],
    captions: list[dict[str, Any]],
    persist_dir: str | None = None,
) -> PaperVectorDB:
    """Compatibility helper for the original command-line workflow."""
    vector_db = PaperVectorDB(persist_dir)
    doc_id = "legacy"
    if texts:
        doc_id = str(texts[0].get("metadata", {}).get("doc_id", doc_id))
    vector_db.index_document(doc_id, texts, captions)
    return vector_db
