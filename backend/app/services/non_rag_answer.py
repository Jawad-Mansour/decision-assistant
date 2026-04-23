"""Generate answers without retrieval (LLM-only)."""

from __future__ import annotations

import logging
from functools import lru_cache

from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class NonRAGAnswerGenerator:
    """Singleton LLM-only answer generator."""

    @staticmethod
    @lru_cache(maxsize=1)
    def get_instance() -> "NonRAGAnswerGenerator":
        return NonRAGAnswerGenerator()

    def __init__(self) -> None:
        self._client = LLMClient.get_instance()
        logger.info("NonRAGAnswerGenerator initialized")

    async def generate(self, user_text: str) -> tuple[str, float, float]:
        """Return answer_text, latency_ms, cost_usd."""
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a professional customer support assistant. Provide a clear, helpful response. "
                    "If you need account-specific details, ask the user to share them via a secure channel. "
                    "Do not invent policy or order facts."
                ),
            },
            {"role": "user", "content": user_text},
        ]
        result = await self._client.complete_chat(messages, temperature=0.4, max_tokens=600)
        return result.text, result.latency_ms, result.usage.estimated_cost_usd
