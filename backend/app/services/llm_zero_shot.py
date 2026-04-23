"""LLM zero-shot ticket priority classification."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Tuple

from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a support triage assistant. Given a customer message, decide if it requires "
    "urgent handling. Reply only with the single word urgent or the single word normal. "
    "No punctuation, no explanation, no other words."
)


def _parse_priority(raw: str) -> str:
    text = raw.strip().lower()
    if not text:
        return "normal"
    if re.search(r"\burgent\b", text):
        return "urgent"
    if re.search(r"\bnormal\b", text):
        return "normal"
    first = text.split()[0] if text.split() else ""
    if first in {"urgent", "normal"}:
        return first
    return "normal"


class LLMZeroShotPredictor:
    """Singleton zero-shot priority predictor using LLMClient."""

    @staticmethod
    @lru_cache(maxsize=1)
    def get_instance() -> "LLMZeroShotPredictor":
        return LLMZeroShotPredictor()

    def __init__(self) -> None:
        self._client = LLMClient.get_instance()
        logger.info("LLMZeroShotPredictor initialized")

    async def predict(self, user_text: str) -> Tuple[str, float, float, int, int]:
        """
        Returns:
            priority ('urgent'|'normal'), latency_ms, cost_usd, prompt_tokens, completion_tokens
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text[:2000]},
        ]
        result = await self._client.complete_chat(messages, temperature=0.0, max_tokens=8)
        priority = _parse_priority(result.text)
        return (
            priority,
            result.latency_ms,
            result.usage.estimated_cost_usd,
            result.usage.prompt_tokens,
            result.usage.completion_tokens,
        )
