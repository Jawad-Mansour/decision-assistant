"""Generate answers using retrieved context + LLM."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from app.schemas.common import RetrievedContext
from app.services.llm_client import LLMClient
from app.services.rag_retriever import RagRetriever
from app.core.paths import get_chroma_path

logger = logging.getLogger(__name__)

# Per retrieved doc: how much raw text the LLM sees (token / context limits).
_PROMPT_PASSAGE_CHARS = 1200
_RESPONSE_SNIPPET_CHARS = 600


def _snippet(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    if max_chars < 2:
        return "…"
    return t[: max_chars - 1] + "…"


def _chroma_cosine_similarity(distance: float | None) -> float | None:
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
        # Direct Chroma check using dynamic path resolution
        try:
            import chromadb
            
            persist_dir = str(get_chroma_path())
            
            if not Path(persist_dir).exists():
                raise FileNotFoundError(f"Directory does not exist: {persist_dir}")
            
            logger.info(f"Checking Chroma at: {persist_dir}")
            client = chromadb.PersistentClient(path=persist_dir)
            collection = client.get_collection("support_conversations")
            vector_count = collection.count()
            
            if vector_count == 0:
                raise FileNotFoundError(f"No vectors found at {persist_dir}")
            
            logger.info(f"RAG ready with {vector_count:,} vectors")
            
        except Exception as e:
            logger.error(f"RAG unavailable: {e}")
            raise FileNotFoundError(f"Vector store is empty or not found. Error: {e}")

        # Get retrieval results
        rows = self._retriever.retrieve(user_text, top_k=top_k)
        logger.info(f"Retrieved {len(rows)} rows for query: {user_text[:50]}")
        
        # Build context block for LLM
        context_block = "\n\n---\n\n".join(
            f"[{idx + 1}] (id={row['id']})\n{_snippet(str(row.get('text') or ''), _PROMPT_PASSAGE_CHARS)}"
            for idx, row in enumerate(rows)
        )
        
        # Build contexts for API response
        contexts = []
        for row in rows:
            ctx = RetrievedContext(
                id=str(row["id"]),
                text=_snippet(str(row.get("text") or ""), _RESPONSE_SNIPPET_CHARS),
                distance=row.get("distance"),
                similarity_score=_chroma_cosine_similarity(row.get("distance")),
            )
            contexts.append(ctx)
        
        logger.info(f"Built {len(contexts)} contexts for response")
        
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