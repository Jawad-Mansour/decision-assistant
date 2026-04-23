from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.schemas.answer import AnswerResponse
from app.schemas.common import PriorityResponse, TextInput


class QueryRequest(TextInput):
    """Unified query request for side-by-side comparison."""

    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of retrieval results to use for RAG branch.",
    )


class QueryResponse(BaseModel):
    """Unified query response: ML/LLM priorities + RAG/non-RAG answers."""

    text: str = Field(..., description="Sanitized user query text.")
    ml_priority: PriorityResponse = Field(..., description="ML priority output.")
    llm_priority: PriorityResponse = Field(..., description="LLM zero-shot priority output.")
    rag_answer: AnswerResponse = Field(..., description="RAG answer output.")
    non_rag_answer: AnswerResponse = Field(..., description="Non-RAG answer output.")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response creation timestamp in UTC.",
    )

