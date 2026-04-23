"""
Vector store service for RAG retrieval.

Uses ChromaDB persistent storage and the local Embedder service to:
- upsert text documents with metadata
- query nearest neighbors for a user question
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import chromadb

from app.core.config import get_settings
from app.services.embedder import Embedder

logger = logging.getLogger(__name__)


class VectorStore:
    """Manage ChromaDB collection lifecycle, indexing, and retrieval."""

    DEFAULT_COLLECTION_NAME = "support_conversations"

    @staticmethod
    @lru_cache(maxsize=1)
    def get_instance(
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_directory: str | None = None,
        model_name: str = Embedder.DEFAULT_MODEL_NAME,
    ) -> "VectorStore":
        """Return a cached vector store instance."""
        return VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            model_name=model_name,
        )

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_directory: str | None = None,
        model_name: str = Embedder.DEFAULT_MODEL_NAME,
    ) -> None:
        if not collection_name or not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string.")

        project_root = Path(__file__).resolve().parents[3]
        configured = get_settings().chroma_persist_directory
        default_persist_dir = (
            Path(configured) if configured else project_root / "data" / "chroma_db"
        )
        self.persist_directory = Path(persist_directory) if persist_directory else default_persist_dir
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name.strip()
        self.embedder = Embedder.get_instance(model_name=model_name)

        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_dimension": int(self.embedder.dimension),
                "embedding_model": self.embedder.model_name,
            },
        )

        self._validate_embedding_dimension()
        logger.info(
            "Initialized vector store collection='%s' path='%s' count=%d",
            self.collection_name,
            self.persist_directory,
            self.count(),
        )

    def _validate_embedding_dimension(self) -> None:
        """
        Validate collection embedding dimension against embedder dimension.

        Chroma metadata is best-effort, so this checks only when the metadata key
        is present and parseable.
        """
        metadata = self.collection.metadata or {}
        raw_dim = metadata.get("embedding_dimension")
        if raw_dim is None:
            return

        try:
            stored_dim = int(raw_dim)
        except (TypeError, ValueError):
            logger.warning("Collection has non-integer embedding_dimension metadata: %s", raw_dim)
            return

        if stored_dim != self.embedder.dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"collection={stored_dim}, embedder={self.embedder.dimension}."
            )

    def count(self) -> int:
        """Return the number of indexed vectors/documents."""
        return int(self.collection.count())

    def upsert_texts(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
        *,
        upsert_batch_size: int = 512,
        embed_batch_size: int = 64,
    ) -> int:
        """
        Embed and upsert documents in batches.

        Returns:
            Number of records processed.
        """
        if upsert_batch_size <= 0:
            raise ValueError("upsert_batch_size must be > 0.")
        if embed_batch_size <= 0:
            raise ValueError("embed_batch_size must be > 0.")
        if len(ids) != len(texts):
            raise ValueError("ids and texts length mismatch.")
        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError("metadatas length must match ids/texts length.")
        if not ids:
            return 0

        total = len(ids)
        for start in range(0, total, upsert_batch_size):
            end = min(start + upsert_batch_size, total)

            batch_ids = [str(x) for x in ids[start:end]]
            batch_texts = [str(x) if x is not None else "" for x in texts[start:end]]
            batch_metadatas = list(metadatas[start:end]) if metadatas is not None else None

            embeddings = self.embedder.encode(batch_texts, batch_size=embed_batch_size).tolist()
            self.collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metadatas,
            )
            logger.info("Upserted vectors: %d/%d", end, total)

        return total

    def query(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-k nearest documents for a query.

        Returns:
            List of retrieval rows with id, text, metadata, distance.
        """
        if not query_text or not query_text.strip():
            raise ValueError("query_text must be a non-empty string.")
        if top_k <= 0:
            raise ValueError("top_k must be > 0.")

        query_embedding = self.embedder.encode_single(query_text).tolist()
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        rows: list[dict[str, Any]] = []
        for idx in range(len(ids)):
            rows.append(
                {
                    "id": ids[idx],
                    "text": docs[idx] if idx < len(docs) else None,
                    "metadata": metas[idx] if idx < len(metas) else None,
                    "distance": float(dists[idx]) if idx < len(dists) else None,
                }
            )
        return rows

