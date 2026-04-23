from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.common import RetrievedContext, TextInput


class AnswerRequest(TextInput):
    """Answer generation request (RAG or non-RAG)."""

    mode: Literal["rag", "non_rag"] = Field(
        "rag",
        description="Answer generation mode.",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of contexts to retrieve when mode='rag'.",
    )


class AnswerResponse(BaseModel):
    """Answer generation response payload."""

    mode: Literal["rag", "non_rag"] = Field(..., description="Answer generation mode.")
    answer_text: str = Field(..., min_length=1, description="Final generated answer.")
    contexts: list[RetrievedContext] = Field(
        default_factory=list,
        description="Retrieved contexts used for answer generation.",
    )
    latency_ms: float = Field(..., ge=0, description="Response time in milliseconds.")
    cost_dollars: float = Field(..., ge=0, description="Estimated request cost in USD.")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response creation timestamp in UTC.",
    )

    @field_validator("answer_text")
    @classmethod
    def sanitize_answer_text(cls, value: str) -> str:
        sanitized = value.strip()
        if not sanitized:
            raise ValueError("answer_text must not be empty")
        return sanitized

