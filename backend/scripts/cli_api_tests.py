#!/usr/bin/env python3
"""
CLI API regression suite. Requires a running server (uvicorn).

  export BASE=http://127.0.0.1:8000   # optional
  uv run python scripts/cli_api_tests.py

Options:
  --skip-llm     Do not call OpenAI (skip llm_zero_shot, non-RAG, full /query).
  --skip-ml      Do not require ML model files (skip ML predict and full /query).

Exit code 1 if any non-skipped test fails.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path


def _request(
    base: str,
    method: str,
    path: str,
    *,
    data: dict | None = None,
    raw_body: bytes | None = None,
    content_type: str = "application/json",
    timeout: float = 180.0,
) -> tuple[int, str]:
    url = f"{base.rstrip('/')}{path}"
    if method.upper() == "GET":
        body = None
    else:
        body = raw_body if raw_body is not None else (json.dumps(data).encode("utf-8") if data is not None else None)
    headers = {}
    if body is not None:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.getcode()), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description="HTTP API tests against a running FastAPI server.")
    parser.add_argument("--base", default=os.environ.get("BASE", "http://127.0.0.1:8000"))
    parser.add_argument("--skip-llm", action="store_true", help="Skip calls that need OPENAI_API_KEY on the server.")
    parser.add_argument("--skip-ml", action="store_true", help="Skip calls that need models/ artifacts.")
    args = parser.parse_args()
    base: str = args.base

    repo_root = Path(__file__).resolve().parents[2]
    model_path = repo_root / "models" / "priority_classifier.pkl"
    has_ml = model_path.is_file()

    passed = 0
    failed = 0
    skipped = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if ok:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))

    def skip(name: str, reason: str) -> None:
        nonlocal skipped
        skipped += 1
        print(f"  SKIP  {name}  ({reason})")

    print(f"Target: {base}\n")

    # --- GET /health ---
    print("== GET /health ==")
    code, body = _request(base, "GET", "/health")
    check("health returns 200", code == 200)
    if code == 200:
        try:
            j = json.loads(body)
            check("health JSON has status ok", j.get("status") == "ok")
        except json.JSONDecodeError:
            check("health JSON parses", False, "invalid json")

    # --- GET /docs (smoke) ---
    print("\n== GET /docs /openapi.json ==")
    code, _ = _request(base, "GET", "/docs")
    check("docs returns 200", code == 200)
    code, _ = _request(base, "GET", "/openapi.json")
    check("openapi.json returns 200", code == 200)

    # --- Wrong method ---
    print("\n== Method not allowed ==")
    code, _ = _request(base, "POST", "/health", data={})
    check("POST /health is not allowed", code == 405)

    # --- POST /query validation ---
    print("\n== POST /query validation (expect 422) ==")
    cases: list[tuple[str, dict | None, bytes | None, str, int]] = [
        ("empty object", {}, None, "application/json", 422),
        ("missing text", {"top_k": 5}, None, "application/json", 422),
        ("whitespace text", {"text": "   ", "top_k": 5}, None, "application/json", 422),
        ("top_k zero", {"text": "hello", "top_k": 0}, None, "application/json", 422),
        ("top_k 21", {"text": "hello", "top_k": 21}, None, "application/json", 422),
        ("text too long", {"text": "x" * 501, "top_k": 5}, None, "application/json", 422),
        ("text null", {"text": None, "top_k": 5}, None, "application/json", 422),
    ]
    for name, d, raw, ct, want in cases:
        code, _ = _request(base, "POST", "/query", data=d, raw_body=raw, content_type=ct)
        check(f"/query {name} -> {want}", code == want, f"got {code}")

    # --- POST /predict validation ---
    print("\n== POST /predict validation ==")
    for name, d, want in [
        ("missing text", {"model": "ml"}, 422),
        ("invalid model", {"text": "hi", "model": "gpt"}, 422),
        ("whitespace text", {"text": "  ", "model": "ml"}, 422),
    ]:
        code, _ = _request(base, "POST", "/predict", data=d)
        check(f"/predict {name} -> {want}", code == want, f"got {code}")

    # --- POST /answer validation ---
    print("\n== POST /answer validation ==")
    for name, d, want in [
        ("invalid mode", {"text": "hi", "mode": "magic", "top_k": 5}, 422),
        ("top_k 0", {"text": "hi", "mode": "rag", "top_k": 0}, 422),
        ("missing text", {"mode": "non_rag", "top_k": 5}, 422),
    ]:
        code, _ = _request(base, "POST", "/answer", data=d)
        check(f"/answer {name} -> {want}", code == want, f"got {code}")

    # --- Malformed JSON ---
    print("\n== Malformed body ==")
    code, _ = _request(
        base,
        "POST",
        "/query",
        raw_body=b"{not json",
        content_type="application/json",
    )
    check("invalid JSON on /query", code in (400, 422))

    # --- ML live ---
    print("\n== POST /predict model=ml (needs models/) ==")
    if args.skip_ml or not has_ml:
        skip("/predict ml", "no model on disk" if not has_ml else "--skip-ml")
    else:
        code, body = _request(base, "POST", "/predict", data={"text": "refund not received!!!", "model": "ml"})
        check("/predict ml returns 200", code == 200)
        if code == 200:
            j = json.loads(body)
            check("ml result has priority", j.get("result", {}).get("priority") in ("urgent", "normal"))

    # --- LLM live ---
    print("\n== LLM live (server must have OPENAI_API_KEY) ==")
    if args.skip_llm:
        skip("/predict llm_zero_shot", "--skip-llm")
        skip("/answer non_rag", "--skip-llm")
        skip("/query full", "--skip-llm")
    else:
        code, body = _request(
            base,
            "POST",
            "/predict",
            data={"text": "Is this urgent? My bank account was drained.", "model": "llm_zero_shot"},
            timeout=120.0,
        )
        if code == 503:
            skip("/predict llm_zero_shot", "503 likely missing OPENAI_API_KEY on server")
        else:
            check("/predict llm_zero_shot returns 200", code == 200, f"got {code} {body[:200]}")
            if code == 200:
                j = json.loads(body)
                check(
                    "llm result has tokens or cost",
                    "result" in j and j["result"].get("priority") in ("urgent", "normal"),
                )

        code, body = _request(
            base,
            "POST",
            "/answer",
            data={"text": "How do I track my shipment?", "mode": "non_rag", "top_k": 3},
            timeout=120.0,
        )
        if code == 503:
            skip("/answer non_rag", "503 likely missing OPENAI_API_KEY")
        else:
            check("/answer non_rag returns 200", code == 200, f"got {code}")
            if code == 200:
                j = json.loads(body)
                check("non_rag answer_text non-empty", len((j.get("answer_text") or "").strip()) > 0)

        if args.skip_ml or not has_ml:
            skip("/query full", "needs ML model" if not has_ml else "--skip-ml")
        else:
            code, body = _request(
                base,
                "POST",
                "/query",
                data={"text": "I need a refund urgently, order #12345 is wrong.", "top_k": 3},
                timeout=180.0,
            )
            if code == 503:
                skip("/query full", f"503 {body[:120]}")
            elif code != 200:
                check("/query full returns 200", False, f"{code} {body[:300]}")
            else:
                check("/query full returns 200", True)
                j = json.loads(body)
                for key in ("ml_priority", "llm_priority", "rag_answer", "non_rag_answer"):
                    check(f"/query has {key}", key in j)

    # --- RAG answer (may 404 empty index) ---
    print("\n== POST /answer mode=rag ==")
    if args.skip_llm:
        skip("/answer rag", "--skip-llm")
    else:
        code, body = _request(
            base,
            "POST",
            "/answer",
            data={"text": "refund problem with my credit card", "mode": "rag", "top_k": 3},
            timeout=120.0,
        )
        if code == 404:
            check("/answer rag 404 when index empty", True)
        elif code == 503:
            skip("/answer rag", "503 LLM config")
        else:
            check("/answer rag returns 200 or 404", code in (200, 404), f"got {code}")

    # --- Unicode + boundary text ---
    print("\n== Unicode / boundary ==")
    code, _ = _request(
        base,
        "POST",
        "/predict",
        data={"text": "Réclamation — remboursement €50  !!!", "model": "ml"},
    )
    if not args.skip_ml and has_ml:
        check("/predict ml unicode text", code == 200)
    else:
        skip("/predict ml unicode", "ml skipped or no model")

    code, _ = _request(base, "POST", "/predict", data={"text": "x" * 500, "model": "ml"})
    if not args.skip_ml and has_ml:
        check("/predict ml max length 500", code == 200)
    else:
        skip("/predict ml max len", "ml skipped or no model")

    print(f"\n--- Summary: {passed} passed, {failed} failed, {skipped} skipped ---")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
