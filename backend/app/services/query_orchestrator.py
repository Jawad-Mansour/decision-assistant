"""Orchestrate ML/LLM priority + RAG/non-RAG answers for /query (no nested HTTP handlers)."""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from app.schemas.answer import AnswerResponse
from app.schemas.common import PriorityResponse
from app.schemas.query import QueryRequest, QueryResponse
from app.services.ml_predictor import MLPredictor
from app.services.non_rag_answer import NonRAGAnswerGenerator
from app.services.rag_answer_generator import RAGAnswerGenerator
from app.services.llm_zero_shot import LLMZeroShotPredictor

logger = logging.getLogger(__name__)


class QueryOrchestrator:
    """Run all four branches with parallel LLM work where safe."""

    @staticmethod
    @lru_cache(maxsize=1)
    def get_instance() -> "QueryOrchestrator":
        return QueryOrchestrator()

    async def run(self, payload: QueryRequest) -> QueryResponse:
        text = payload.text
        top_k = payload.top_k

        ml_predictor = MLPredictor.get_instance()
        llm_zero = LLMZeroShotPredictor.get_instance()
        rag_gen = RAGAnswerGenerator.get_instance()
        non_rag_gen = NonRAGAnswerGenerator.get_instance()

        ml_task = asyncio.create_task(ml_predictor.predict_with_metrics(text))
        llm_task = asyncio.create_task(llm_zero.predict(text))
        rag_task = asyncio.create_task(rag_gen.generate(text, top_k))
        non_rag_task = asyncio.create_task(non_rag_gen.generate(text))

        ml_priority: PriorityResponse
        llm_priority: PriorityResponse
        rag_answer: AnswerResponse
        non_rag_answer: AnswerResponse

        try:
            m_priority, m_conf, m_lat, m_cost = await ml_task
            ml_priority = PriorityResponse(
                priority=m_priority,
                confidence=m_conf,
                latency_ms=m_lat,
                cost_dollars=m_cost,
                prompt_tokens=None,
                completion_tokens=None,
            )
        except Exception:
            logger.exception("ML prediction failed")
            raise

        try:
            z_priority, z_lat, z_cost, z_pt, z_ct = await llm_task
            llm_priority = PriorityResponse(
                priority=z_priority,
                confidence=None,
                latency_ms=z_lat,
                cost_dollars=z_cost,
                prompt_tokens=z_pt,
                completion_tokens=z_ct,
            )
        except Exception:
            logger.exception("LLM zero-shot failed")
            raise

        try:
            r_text, r_contexts, r_lat, r_cost = await rag_task
            rag_answer = AnswerResponse(
                mode="rag",
                answer_text=r_text,
                contexts=r_contexts,
                latency_ms=r_lat,
                cost_dollars=r_cost,
            )
        except FileNotFoundError:
            rag_answer = AnswerResponse(
                mode="rag",
                answer_text="Retrieval index is not available. Run vector ingestion, then retry.",
                contexts=[],
                latency_ms=0.0,
                cost_dollars=0.0,
            )
        except Exception:
            logger.exception("RAG answer failed")
            raise

        try:
            n_text, n_lat, n_cost = await non_rag_task
            non_rag_answer = AnswerResponse(
                mode="non_rag",
                answer_text=n_text,
                contexts=[],
                latency_ms=n_lat,
                cost_dollars=n_cost,
            )
        except Exception:
            logger.exception("Non-RAG answer failed")
            raise

        return QueryResponse(
            text=text,
            ml_priority=ml_priority,
            llm_priority=llm_priority,
            rag_answer=rag_answer,
            non_rag_answer=non_rag_answer,
        )
