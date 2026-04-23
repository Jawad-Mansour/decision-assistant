"""
Embedding generation for RAG retrieval.

This module provides a lightweight singleton wrapper around
`sentence-transformers` for generating dense vector embeddings.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Handle text embedding generation for RAG indexing and retrieval."""

    DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

    @staticmethod
    @lru_cache(maxsize=1)
    def get_instance(model_name: str = DEFAULT_MODEL_NAME) -> "Embedder":
        """
        Return a cached embedder instance.

        Notes:
        - `maxsize=1` keeps one process-wide singleton.
        - If a different model name is passed later, the cache keeps only
          the latest requested model instance.
        """
        return Embedder(model_name=model_name)

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        self.model_name = model_name.strip()
        self.model = SentenceTransformer(self.model_name)
        # sentence-transformers renamed this API in newer versions.
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = int(self.model.get_embedding_dimension())
        else:
            self.dimension = int(self.model.get_sentence_embedding_dimension())
        logger.info(
            "Initialized embedder model='%s' dimension=%d",
            self.model_name,
            self.dimension,
        )

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """
        Generate embeddings for a sequence of texts.

        Args:
            texts: Input texts to encode.
            batch_size: Batch size for transformer inference.

        Returns:
            np.ndarray of shape (n_texts, embedding_dim) with dtype float32.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        normalized_texts = [str(t) if t is not None else "" for t in texts]
        embeddings = self.model.encode(
            normalized_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a single text.

        Returns:
            np.ndarray of shape (embedding_dim,) and dtype float32.
        """
        return self.encode([text])[0]

