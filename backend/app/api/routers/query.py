from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Response, status

from app.schemas.common import HealthResponse
from app.schemas.query import QueryRequest, QueryResponse
from app.services.query_orchestrator import QueryOrchestrator

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check() -> HealthResponse:
    """Basic health endpoint for readiness checks."""
    logger.info("GET /health")
    return HealthResponse(status="ok")


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def run_full_query(payload: QueryRequest, response: Response) -> QueryResponse:
    """Return side-by-side ML/LLM priority and RAG/non-RAG answer outputs."""
    response.headers["X-RateLimit-Limit"] = "100"
    response.headers["X-RateLimit-Remaining"] = "99"
    logger.info("POST /query top_k=%d text_length=%d", payload.top_k, len(payload.text))

    try:
        orchestrator = QueryOrchestrator.get_instance()
        return await orchestrator.run(payload)
    except ValueError as exc:
        logger.warning("Query configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected query endpoint error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while running query pipeline.",
        ) from exc
