"""
RAG retrieval service.

Thin service wrapper around VectorStore to keep query retrieval logic
in one place for future API routes.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RagRetriever:
    """Retrieve relevant conversation chunks for a user query."""

    @staticmethod
    @lru_cache(maxsize=1)
    def get_instance() -> "RagRetriever":
        """Return a cached retriever instance."""
        return RagRetriever()

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self.vector_store = vector_store or VectorStore.get_instance()
        logger.info(
            "Initialized RagRetriever collection='%s' vectors=%d",
            self.vector_store.collection_name,
            self.vector_store.count(),
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve top-k nearest context chunks."""
        return self.vector_store.query(query_text=query, top_k=top_k, where=where)

