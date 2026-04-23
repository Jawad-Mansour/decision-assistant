from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator

MIN_TEXT_LENGTH = 1
MAX_TEXT_LENGTH = 500


class TextInput(BaseModel):
    """Shared text input with strict sanitization and length checks."""

    text: str = Field(
        ...,
        min_length=MIN_TEXT_LENGTH,
        max_length=MAX_TEXT_LENGTH,
        description="User input text.",
    )

    @field_validator("text")
    @classmethod
    def sanitize_text(cls, value: str) -> str:
        sanitized = value.strip()
        if not sanitized:
            raise ValueError("text must not be empty after trimming whitespace")
        if len(sanitized) > MAX_TEXT_LENGTH:
            raise ValueError(f"text length must be <= {MAX_TEXT_LENGTH} characters")
        return sanitized


class PriorityResponse(BaseModel):
    """Priority prediction response details."""

    priority: str = Field(..., description="'urgent' or 'normal'")
    confidence: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Confidence score (0-1), only for ML outputs.",
    )
    latency_ms: float = Field(..., ge=0, description="Response time in milliseconds")
    cost_dollars: float = Field(
        ...,
        ge=0,
        description="USD estimated from list-price token rates (0 for on-box ML).",
    )
    prompt_tokens: Optional[int] = Field(
        None,
        ge=0,
        description="Prompt tokens from the provider (LLM paths only).",
    )
    completion_tokens: Optional[int] = Field(
        None,
        ge=0,
        description="Completion tokens from the provider (LLM paths only).",
    )

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"urgent", "normal"}:
            raise ValueError('priority must be "urgent" or "normal"')
        return normalized


class RetrievedContext(BaseModel):
    """Retrieved vector store context item."""

    id: str = Field(..., description="Vector document identifier.")
    text: str = Field(
        ...,
        description="Snippet of retrieved text for display (full row may be longer in the index).",
    )
    distance: Optional[float] = Field(
        None,
        ge=0,
        description="Chroma cosine distance (1 - cosine similarity); lower is closer.",
    )
    similarity_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Cosine similarity in [0,1] derived from Chroma cosine distance.",
    )


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = Field(..., description="Service status, usually 'ok'.")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Health check UTC timestamp.",
    )

