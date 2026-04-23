"""Multi-provider async LLM client with retries, usage, and cost estimates."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# List-price USD per 1M tokens (input / output) from provider pricing pages.
# Billable cost is computed from actual token counts returned by the API (honest usage);
# rates here are estimates until reconciled with monthly invoices.
_MODEL_COSTS_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "gemini-2.0-flash": (0.10, 0.40),
}


def _estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    inp_rate, out_rate = _MODEL_COSTS_PER_1M.get(model, (0.50, 1.50))
    return (prompt_tokens / 1_000_000.0) * inp_rate + (completion_tokens / 1_000_000.0) * out_rate


@dataclass(frozen=True)
class LLMUsage:
    """Token usage and estimated spend."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float


@dataclass(frozen=True)
class LLMCompletionResult:
    """Normalized completion output."""

    text: str
    usage: LLMUsage
    provider: str
    model: str
    latency_ms: float


class LLMClient:
    """Singleton LLM gateway: OpenAI, Groq (OpenAI-compatible), or Gemini REST."""

    @staticmethod
    @lru_cache(maxsize=1)
    def get_instance() -> "LLMClient":
        return LLMClient()

    def __init__(self) -> None:
        self._settings = get_settings()
        self._http = httpx.AsyncClient(timeout=120.0)
        self._openai_style_client: AsyncOpenAI | None = None
        self._openai_style_label: str | None = None
        logger.info("LLMClient initialized provider=%s", self._settings.llm_provider)

    async def aclose(self) -> None:
        if self._openai_style_client is not None:
            await self._openai_style_client.close()
            self._openai_style_client = None
        await self._http.aclose()

    def _openai_compatible_client(self) -> tuple[AsyncOpenAI, str, str]:
        s = self._settings
        if s.llm_provider == "openai":
            if not s.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            if self._openai_style_client is None or self._openai_style_label != "openai":
                self._openai_style_client = AsyncOpenAI(api_key=s.openai_api_key)
                self._openai_style_label = "openai"
            return self._openai_style_client, s.openai_model, "openai"
        if s.llm_provider == "groq":
            if not s.groq_api_key:
                raise ValueError("GROQ_API_KEY is not set.")
            if self._openai_style_client is None or self._openai_style_label != "groq":
                self._openai_style_client = AsyncOpenAI(
                    api_key=s.groq_api_key,
                    base_url=s.groq_base_url,
                )
                self._openai_style_label = "groq"
            return self._openai_style_client, s.groq_model, "groq"
        raise ValueError(f"OpenAI-compatible client not used for provider={s.llm_provider}")

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
    )
    async def complete_chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> LLMCompletionResult:
        """Run a chat completion for the configured provider."""
        s = self._settings
        start = time.perf_counter()

        if s.llm_provider in {"openai", "groq"}:
            client, model, provider_label = self._openai_compatible_client()
            response = await client.chat.completions.create(
                model=model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = (response.choices[0].message.content or "").strip()
            usage_obj = response.usage
            prompt_tokens = int(usage_obj.prompt_tokens or 0) if usage_obj else 0
            completion_tokens = int(usage_obj.completion_tokens or 0) if usage_obj else 0
            total = int(usage_obj.total_tokens or prompt_tokens + completion_tokens) if usage_obj else 0
            cost = _estimate_cost_usd(model, prompt_tokens, completion_tokens)
            latency_ms = (time.perf_counter() - start) * 1000.0
            return LLMCompletionResult(
                text=text,
                usage=LLMUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total,
                    estimated_cost_usd=cost,
                ),
                provider=provider_label,
                model=model,
                latency_ms=latency_ms,
            )

        if s.llm_provider == "gemini":
            if not s.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is not set.")
            model = s.gemini_model
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={s.gemini_api_key}"
            )
            parts: list[dict[str, str]] = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                prefix = "User" if role == "user" else "Assistant"
                parts.append({"text": f"{prefix}: {content}\n"})
            body: dict[str, Any] = {
                "contents": [{"role": "user", "parts": [{"text": "".join(p["text"] for p in parts)}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
            }
            resp = await self._http.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
            text = ""
            try:
                candidates = data.get("candidates") or []
                if candidates:
                    parts_out = candidates[0].get("content", {}).get("parts") or []
                    if parts_out:
                        text = str(parts_out[0].get("text", "")).strip()
            except (KeyError, IndexError, TypeError):
                text = ""

            usage_meta = data.get("usageMetadata") or {}
            prompt_tokens = int(usage_meta.get("promptTokenCount", 0))
            completion_tokens = int(usage_meta.get("candidatesTokenCount", 0))
            total = int(usage_meta.get("totalTokenCount", prompt_tokens + completion_tokens))
            cost = _estimate_cost_usd(model, prompt_tokens, completion_tokens)
            latency_ms = (time.perf_counter() - start) * 1000.0
            return LLMCompletionResult(
                text=text,
                usage=LLMUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total,
                    estimated_cost_usd=cost,
                ),
                provider="gemini",
                model=model,
                latency_ms=latency_ms,
            )

        raise ValueError(f"Unsupported LLM_PROVIDER: {s.llm_provider}")
