"""HTTP smoke checks against a running API (default http://127.0.0.1:8000).

Usage:
    uv run python scripts/smoke_api.py
    BACKEND_URL=http://localhost:8000 uv run python scripts/smoke_api.py

Set RUN_LIVE_QUERY=1 to also POST /query (requires OPENAI_API_KEY and optional Chroma data).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def _get(url: str, timeout: float = 30.0) -> tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), resp.read().decode("utf-8", errors="replace")


def _post_json(url: str, payload: dict, timeout: float = 120.0) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), resp.read().decode("utf-8", errors="replace")


def main() -> int:
    base = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        code, body = _get(f"{base}/health")
    except urllib.error.URLError as exc:
        print(f"ERROR: cannot reach {base}/health: {exc}", file=sys.stderr)
        return 1
    if code != 200:
        print(f"ERROR: /health returned {code}", file=sys.stderr)
        return 1
    print("OK /health", body[:200])

    if os.environ.get("RUN_LIVE_QUERY") == "1":
        try:
            qcode, qbody = _post_json(
                f"{base}/query",
                {"text": "My refund has not arrived and I need help urgently.", "top_k": 3},
            )
        except urllib.error.HTTPError as exc:
            print(f"ERROR: /query HTTP {exc.code}: {exc.read().decode(errors='replace')}", file=sys.stderr)
            return 1
        except urllib.error.URLError as exc:
            print(f"ERROR: /query failed: {exc}", file=sys.stderr)
            return 1
        if qcode != 200:
            print(f"ERROR: /query returned {qcode}: {qbody[:500]}", file=sys.stderr)
            return 1
        print("OK /query", qbody[:300], "...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
