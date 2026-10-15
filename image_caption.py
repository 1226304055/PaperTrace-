"""Cached VLM descriptions for figures extracted from papers."""

from __future__ import annotations

import base64
import threading
import time
from pathlib import Path
from typing import Any

from config import Settings, settings
from storage import MetadataStore


SYSTEM_PROMPT = """你负责为论文图表生成可检索的描述。请说明：
1. 图表类型；
2. 关键对象、数据流或趋势；
3. 重要坐标轴、图例和数值；
4. 该图支持的主要结论。
只描述图中可见内容，不推测未展示的信息。使用与论文图表一致的语言。"""


def encode_image(image_path: str | Path) -> str:
    with Path(image_path).open("rb") as stream:
        return base64.b64encode(stream.read()).decode("utf-8")


def media_type(image_path: str | Path) -> str:
    suffix = Path(image_path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/png")


class ImageCaptioner:
    def __init__(
        self,
        store: MetadataStore,
        app_settings: Settings = settings,
        client: Any | None = None,
    ):
        self.store = store
        self.settings = app_settings
        self._client = client
        self._client_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.settings.api_key or self._client)

    def _get_client(self) -> Any:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    from openai import OpenAI

                    self._client = OpenAI(
                        api_key=self.settings.api_key,
                        base_url=self.settings.base_url,
                        timeout=60,
                    )
        return self._client

    def generate_caption(self, image: dict[str, Any]) -> str | None:
        cache_key = f"caption:{self.settings.vision_model}:{image['image_hash']}"
        if cached := self.store.cache_get(cache_key):
            return cached
        if not self.enabled:
            return None

        image_path = image["path"]
        payload = f"data:{media_type(image_path)};base64,{encode_image(image_path)}"
        error: Exception | None = None
        for attempt in range(self.settings.retry_attempts):
            try:
                response = self._get_client().chat.completions.create(
                    model=self.settings.vision_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "请描述这张论文图表。"},
                                {"type": "image_url", "image_url": {"url": payload}},
                            ],
                        },
                    ],
                    max_tokens=self.settings.vision_max_tokens,
                    temperature=0.1,
                )
                caption = response.choices[0].message.content.strip()
                if caption:
                    self.store.cache_set(cache_key, caption)
                    return caption
            except Exception as exc:  # retry boundary for remote model failures
                error = exc
                if attempt + 1 < self.settings.retry_attempts:
                    time.sleep(self.settings.retry_backoff_seconds * (2**attempt))
        if error:
            raise RuntimeError(f"图表描述生成失败: {error}") from error
        return None

    def caption_images(self, images: list[dict[str, Any]]) -> list[dict[str, Any]]:
        captions: list[dict[str, Any]] = []
        for image in images:
            caption = self.generate_caption(image)
            if not caption:
                continue
            metadata = dict(image["metadata"])
            metadata.update(
                {
                    "content_type": "image_caption",
                    "source_image_path": image["path"],
                    "image_hash": image["image_hash"],
                }
            )
            captions.append({"text": caption, "metadata": metadata})
        return captions


def batch_caption(
    image_paths: list[dict[str, Any]],
    output_file: str | None = None,
) -> list[dict[str, Any]]:
    """Compatibility helper; application code uses ``ImageCaptioner`` directly."""
    del output_file
    store = MetadataStore(settings.sqlite_db_path)
    return ImageCaptioner(store).caption_images(image_paths)
