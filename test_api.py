"""Small command-line smoke client for a running PaperTrace server."""

from __future__ import annotations

import argparse
import json
from urllib import request


def call(method: str, url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["health", "index", "ask"])
    parser.add_argument("value", nargs="?")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    if args.action == "health":
        result = call("GET", f"{args.base_url}/api/health")
    elif args.action == "index":
        result = call(
            "POST",
            f"{args.base_url}/api/index",
            {"pdf_dir": args.value or "./pdfs", "force": False},
        )
    else:
        result = call(
            "POST",
            f"{args.base_url}/api/ask",
            {"message": args.value or "这篇论文的主要贡献是什么？", "k": 5},
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
