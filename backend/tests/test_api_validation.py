"""HTTP validation tests (no live OpenAI; no dependency on models for most cases)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_get_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_get_openapi() -> None:
    assert client.get("/openapi.json").status_code == 200


def test_post_health_method_not_allowed() -> None:
    assert client.post("/health", json={}).status_code == 405


@pytest.mark.parametrize(
    ("path", "payload", "expected"),
    [
        ("/query", {}, 422),
        ("/query", {"top_k": 5}, 422),
        ("/query", {"text": "", "top_k": 5}, 422),
        ("/query", {"text": "   ", "top_k": 5}, 422),
        ("/query", {"text": "hi", "top_k": 0}, 422),
        ("/query", {"text": "hi", "top_k": 22}, 422),
        ("/query", {"text": "x" * 501, "top_k": 5}, 422),
    ],
)
def test_query_validation(path: str, payload: dict, expected: int) -> None:
    r = client.post(path, json=payload)
    assert r.status_code == expected, r.text


def test_query_null_text_raw_json() -> None:
    r = client.post(
        "/query",
        content=json.dumps({"text": None, "top_k": 5}),
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"model": "ml"}, 422),
        ({"text": "hi", "model": "unknown"}, 422),
        ({"text": "  ", "model": "ml"}, 422),
    ],
)
def test_predict_validation(payload: dict, expected: int) -> None:
    r = client.post("/predict", json=payload)
    assert r.status_code == expected


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"text": "hi", "mode": "invalid", "top_k": 5}, 422),
        ({"text": "hi", "mode": "rag", "top_k": 0}, 422),
        ({"mode": "non_rag", "top_k": 5}, 422),
    ],
)
def test_answer_validation(payload: dict, expected: int) -> None:
    r = client.post("/answer", json=payload)
    assert r.status_code == expected


def test_malformed_json_query() -> None:
    r = client.post(
        "/query",
        content="{not-json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code in (400, 422)


@pytest.mark.skipif(
    not Path(__file__).resolve().parents[2].joinpath("models", "priority_classifier.pkl").is_file(),
    reason="ML model artifact not present",
)
def test_predict_ml_ok_when_model_exists() -> None:
    r = client.post("/predict", json={"text": "refund help urgent!!!", "model": "ml"})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "ml"
    assert body["result"]["priority"] in ("urgent", "normal")
    assert body["result"]["cost_dollars"] == 0.0
