from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Response, status

from app.schemas.common import PriorityResponse
from app.schemas.predict import PredictRequest, PredictResponse
from app.services import MLPredictor
from app.services.llm_zero_shot import LLMZeroShotPredictor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict_priority(payload: PredictRequest, response: Response) -> PredictResponse:
    """Predict ticket priority via ML or LLM zero-shot."""
    response.headers["X-RateLimit-Limit"] = "100"
    response.headers["X-RateLimit-Remaining"] = "99"

    logger.info("POST /predict model=%s text_length=%d", payload.model, len(payload.text))

    try:
        if payload.model == "ml":
            predictor = MLPredictor.get_instance()
            priority, confidence, latency_ms, cost_dollars = await predictor.predict_with_metrics(payload.text)
            result = PriorityResponse(
                priority=priority,
                confidence=confidence,
                latency_ms=latency_ms,
                cost_dollars=cost_dollars,
                prompt_tokens=None,
                completion_tokens=None,
            )
            return PredictResponse(model="ml", result=result)

        zero = LLMZeroShotPredictor.get_instance()
        priority, latency_ms, cost_dollars, pt, ct = await zero.predict(payload.text)
        result = PriorityResponse(
            priority=priority,
            confidence=None,
            latency_ms=latency_ms,
            cost_dollars=cost_dollars,
            prompt_tokens=pt,
            completion_tokens=ct,
        )
        return PredictResponse(model="llm_zero_shot", result=result)

    except FileNotFoundError as exc:
        logger.exception("Predict service dependency missing: %s", exc)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning("Predict configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected predict error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while predicting priority.",
        ) from exc
