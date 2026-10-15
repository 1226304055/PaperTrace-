"""Application configuration for the local paper evidence assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Allows dependency-light unit tests.
    def load_dotenv(*args, **kwargs):  # type: ignore[no-untyped-def]
        return False


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _path_from_env(name: str, default: str) -> Path:
    raw = Path(os.getenv(name, default)).expanduser()
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def _int_from_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float_from_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded once from environment variables."""

    project_root: Path = PROJECT_ROOT
    pdf_dir: Path = _path_from_env("PDF_DIR", "./pdfs")
    data_dir: Path = _path_from_env("DATA_DIR", "./chroma_db")
    chroma_db_dir: Path = _path_from_env("CHROMA_DB_DIR", "./chroma_db/vectors")
    images_dir: Path = _path_from_env("IMAGES_DIR", "./images")
    sqlite_db_path: Path = _path_from_env("SQLITE_DB_PATH", "./chroma_db/papertrace.db")
    embedding_cache_dir: Path = _path_from_env(
        "EMBEDDING_CACHE_DIR", "./chroma_db/models"
    )

    chunk_size: int = _int_from_env("CHUNK_SIZE", 900)
    chunk_overlap: int = _int_from_env("CHUNK_OVERLAP", 120)
    min_image_bytes: int = _int_from_env("MIN_IMAGE_BYTES", 5000)
    max_images_per_pdf: int = _int_from_env("MAX_IMAGES_PER_PDF", 30)
    max_upload_mb: int = _int_from_env("MAX_UPLOAD_MB", 50)

    api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    base_url: str = os.getenv(
        "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    chat_model: str = os.getenv("LLM_MODEL", "qwen-plus")
    vision_model: str = os.getenv("VLM_MODEL", "qwen-vl-max")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
    )

    top_k: int = _int_from_env("TOP_K", 5)
    retrieval_candidates: int = _int_from_env("RETRIEVAL_CANDIDATES", 20)
    lexical_scan_limit: int = _int_from_env("LEXICAL_SCAN_LIMIT", 5000)
    history_turns: int = _int_from_env("HISTORY_TURNS", 4)
    max_context_chars: int = _int_from_env("MAX_CONTEXT_CHARS", 12000)
    max_images_per_query: int = _int_from_env("MAX_IMAGES_PER_QUERY", 3)
    retry_attempts: int = _int_from_env("RETRY_ATTEMPTS", 3)
    retry_backoff_seconds: float = _float_from_env("RETRY_BACKOFF_SECONDS", 1.0)
    caption_workers: int = _int_from_env("CAPTION_WORKERS", 4)
    vision_max_tokens: int = _int_from_env("VISION_MAX_TOKENS", 280)

    def ensure_directories(self) -> None:
        for directory in (
            self.pdf_dir,
            self.data_dir,
            self.chroma_db_dir,
            self.images_dir,
            self.embedding_cache_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()

# Backwards-compatible aliases for small scripts that imported the old constants.
PDF_DIR = str(settings.pdf_dir)
CHROMA_DB_DIR = str(settings.chroma_db_dir)
IMAGES_DIR = str(settings.images_dir)
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
LLM_API_KEY = settings.api_key
LLM_BASE_URL = settings.base_url
LLM_MODEL = settings.chat_model
TOP_K = settings.top_k
