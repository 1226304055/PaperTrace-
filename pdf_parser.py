"""PDF parsing with deterministic IDs and page-level provenance."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import settings


@dataclass
class ParsedPDF:
    doc_id: str
    file_hash: str
    source_path: str
    filename: str
    page_count: int
    text_chunks: list[dict[str, Any]]
    images: list[dict[str, Any]]


def file_sha256(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text on paragraph/sentence boundaries while preserving overlap."""
    cleaned = re.sub(r"[ \t]+", " ", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if not cleaned:
        return []

    units = [unit.strip() for unit in re.split(r"(?<=[。！？.!?])\s+|\n+", cleaned)]
    units = [unit for unit in units if unit]
    chunks: list[str] = []
    current = ""

    for unit in units:
        candidate = f"{current}\n{unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = (current[-overlap:] + "\n" + unit).strip()
        else:
            start = 0
            while start < len(unit):
                chunks.append(unit[start : start + chunk_size])
                start += max(1, chunk_size - overlap)
            current = ""

        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = current[max(1, chunk_size - overlap) :]

    if current:
        chunks.append(current)
    return chunks


def parse_pdf(
    pdf_path: str | Path,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    images_dir: str | Path | None = None,
) -> ParsedPDF:
    """Extract page-aware text chunks and meaningful embedded images."""
    try:
        import pymupdf as fitz
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("缺少 PyMuPDF，请先安装 requirements.txt") from exc

    path = Path(pdf_path).resolve()
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"仅支持 PDF 文件: {path.name}")

    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    digest = file_sha256(path)
    doc_id = digest[:20]
    image_root = Path(images_dir or settings.images_dir) / doc_id
    image_root.mkdir(parents=True, exist_ok=True)

    text_chunks: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    seen_image_hashes: set[str] = set()
    document = fitz.open(path)
    page_count = len(document)
    try:
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            page_text = page.get_text("text").strip()
            for local_index, chunk in enumerate(
                split_text(page_text, chunk_size, chunk_overlap)
            ):
                text_chunks.append(
                    {
                        "text": chunk,
                        "metadata": {
                            "doc_id": doc_id,
                            "source": path.name,
                            "source_path": str(path),
                            "page": page_number,
                            "content_type": "text",
                            "chunk_index": len(text_chunks),
                            "page_chunk_index": local_index,
                        },
                    }
                )

            if len(images) >= settings.max_images_per_pdf:
                continue
            for image_index, image_info in enumerate(page.get_images(full=True)):
                if len(images) >= settings.max_images_per_pdf:
                    break
                extracted = document.extract_image(image_info[0])
                image_bytes = extracted["image"]
                width = int(extracted.get("width", 0))
                height = int(extracted.get("height", 0))
                if (
                    len(image_bytes) < settings.min_image_bytes
                    or width < 80
                    or height < 80
                ):
                    continue
                image_hash = hashlib.sha256(image_bytes).hexdigest()
                if image_hash in seen_image_hashes:
                    continue
                seen_image_hashes.add(image_hash)
                extension = extracted.get("ext", "png")
                image_path = image_root / (
                    f"p{page_number:03d}_{image_index + 1:02d}_{image_hash[:10]}.{extension}"
                )
                if not image_path.exists():
                    image_path.write_bytes(image_bytes)
                images.append(
                    {
                        "path": str(image_path),
                        "image_hash": image_hash,
                        "metadata": {
                            "doc_id": doc_id,
                            "source": path.name,
                            "source_path": str(path),
                            "page": page_number,
                            "content_type": "image",
                        },
                    }
                )
    finally:
        document.close()

    return ParsedPDF(
        doc_id=doc_id,
        file_hash=digest,
        source_path=str(path),
        filename=path.name,
        page_count=page_count,
        text_chunks=text_chunks,
        images=images,
    )


def batch_parse(pdf_dir: str | Path | None = None) -> list[ParsedPDF]:
    directory = Path(pdf_dir or settings.pdf_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return [parse_pdf(path) for path in sorted(directory.glob("*.pdf"))]


if __name__ == "__main__":
    papers = batch_parse()
    print(
        f"Parsed {len(papers)} papers, "
        f"{sum(len(item.text_chunks) for item in papers)} chunks, "
        f"{sum(len(item.images) for item in papers)} images"
    )
