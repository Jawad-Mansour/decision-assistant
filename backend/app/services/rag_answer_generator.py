"""Generate answers using retrieved context + LLM."""

from __future__ import annotations

import logging
from functools import lru_cache
from app.schemas.common import RetrievedContext
from app.services.llm_client import LLMClient
from app.services.rag_retriever import RagRetriever
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Per retrieved doc: how much raw text the LLM sees (token / context limits).
_PROMPT_PASSAGE_CHARS = 1200
# Per retrieved doc: how much we return in API `contexts` (source panel / curl) — full DB rows can be very long threads.
_RESPONSE_SNIPPET_CHARS = 600


def _snippet(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    if max_chars < 2:
        return "…"
    return t[: max_chars - 1] + "…"


def _chroma_cosine_similarity(distance: float | None) -> float | None:
    """Chroma ``cosine`` space uses distance = 1 - cosine_similarity."""
    if distance is None:
        return None
    sim = 1.0 - float(distance)
    return max(0.0, min(1.0, sim))


class RAGAnswerGenerator:
    """Retrieve top-k chunks then ask LLM to answer grounded in context."""

    @staticmethod
    @lru_cache(maxsize=1)
    def get_instance() -> "RAGAnswerGenerator":
        return RAGAnswerGenerator()

    def __init__(self) -> None:
        self._client = LLMClient.get_instance()
        self._retriever = RagRetriever.get_instance()
        logger.info("RAGAnswerGenerator initialized")

    async def generate(self, user_text: str, top_k: int) -> tuple[str, list[RetrievedContext], float, float]:
        """
        Returns:
            answer_text, contexts, latency_ms, cost_usd
        """
        store = VectorStore.get_instance()
        if store.count() <= 0:
            raise FileNotFoundError("Vector store is empty. Run ingestion first.")

        rows = self._retriever.retrieve(user_text, top_k=top_k)

        context_block = "\n\n---\n\n".join(
            f"[{idx + 1}] (id={row['id']})\n"
            f"{_snippet(str(row.get('text') or ''), _PROMPT_PASSAGE_CHARS)}"
            for idx, row in enumerate(rows)
        )

        contexts = [
            RetrievedContext(
                id=str(row["id"]),
                text=_snippet(str(row.get("text") or ""), _RESPONSE_SNIPPET_CHARS),
                distance=row.get("distance"),
                similarity_score=_chroma_cosine_similarity(row.get("distance")),
            )
            for row in rows
        ]
        if not context_block.strip():
            context_block = "(No retrieved passages.)"

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful customer support assistant. Use ONLY the provided "
                    "reference passages to answer. If the passages are insufficient, say what is missing "
                    "and suggest safe next steps. Be concise and professional."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{user_text}\n\nReference passages:\n{context_block}",
            },
        ]

        result = await self._client.complete_chat(messages, temperature=0.3, max_tokens=600)
        return result.text, contexts, result.latency_ms, result.usage.estimated_cost_usd
