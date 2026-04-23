from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Response, status

from app.schemas.answer import AnswerRequest, AnswerResponse
from app.services.non_rag_answer import NonRAGAnswerGenerator
from app.services.rag_answer_generator import RAGAnswerGenerator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/answer", tags=["answer"])


@router.post("", response_model=AnswerResponse, status_code=status.HTTP_200_OK)
async def generate_answer(payload: AnswerRequest, response: Response) -> AnswerResponse:
    """Generate answer via RAG (retrieve + LLM) or non-RAG (LLM-only)."""
    response.headers["X-RateLimit-Limit"] = "100"
    response.headers["X-RateLimit-Remaining"] = "99"

    logger.info("POST /answer mode=%s top_k=%d text_length=%d", payload.mode, payload.top_k, len(payload.text))

    try:
        if payload.mode == "rag":
            rag = RAGAnswerGenerator.get_instance()
            try:
                answer_text, contexts, latency_ms, cost_dollars = await rag.generate(
                    payload.text, payload.top_k
                )
            except FileNotFoundError as exc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(exc),
                ) from exc
            return AnswerResponse(
                mode="rag",
                answer_text=answer_text,
                contexts=contexts,
                latency_ms=latency_ms,
                cost_dollars=cost_dollars,
            )

        gen = NonRAGAnswerGenerator.get_instance()
        answer_text, latency_ms, cost_dollars = await gen.generate(payload.text)
        return AnswerResponse(
            mode="non_rag",
            answer_text=answer_text,
            contexts=[],
            latency_ms=latency_ms,
            cost_dollars=cost_dollars,
        )

    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("Answer LLM configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected answer error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while generating answer.",
        ) from exc
