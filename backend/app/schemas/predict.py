from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import PriorityResponse, TextInput


class PredictRequest(TextInput):
    """Priority prediction input request."""

    model: Literal["ml", "llm_zero_shot"] = Field(
        "ml",
        description="Prediction path to use.",
    )


class PredictResponse(BaseModel):
    """Priority prediction API response."""

    model: Literal["ml", "llm_zero_shot"] = Field(..., description="Model used.")
    result: PriorityResponse = Field(..., description="Priority prediction output.")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response creation timestamp in UTC.",
    )

