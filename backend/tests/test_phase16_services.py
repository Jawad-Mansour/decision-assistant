"""
Phase 16 smoke test script.

This script validates:
1) Embedder initialization and encoding behavior
2) VectorStore upsert + query flow
3) RagRetriever retrieval flow

Run:
    uv run python tests/test_phase16_services.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.services import Embedder, RagRetriever, VectorStore


def run_phase16_smoke_test() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

    print("1) Testing Embedder...")
    embedder = Embedder.get_instance()
    vec = embedder.encode_single("my refund is not showing up")
    batch = embedder.encode(
        ["refund not showing", "where is my money", "the weather is nice"],
        batch_size=2,
    )

    assert vec.shape == (embedder.dimension,), "Single embedding shape mismatch."
    assert batch.shape == (3, embedder.dimension), "Batch embedding shape mismatch."
    assert 0.95 <= float(np.linalg.norm(vec)) <= 1.05, "Embedding should be normalized."
    assert float(np.dot(batch[0], batch[1])) > float(
        np.dot(batch[0], batch[2])
    ), "Semantic similarity check failed."
    print("   Embedder OK")

    print("2) Testing VectorStore...")
    project_root = Path(__file__).resolve().parents[1]
    test_persist_dir = project_root / "data" / "chroma_db_test"

    vector_store = VectorStore(
        collection_name="support_conversations_phase16_smoke",
        persist_directory=str(test_persist_dir),
    )

    upserted = vector_store.upsert_texts(
        ids=["phase16_test_1", "phase16_test_2", "phase16_test_3"],
        texts=[
            "my refund is not showing",
            "phone is broken and not working",
            "today is sunny and beautiful",
        ],
        metadatas=[
            {"tweet_id": "phase16_test_1", "type": "refund"},
            {"tweet_id": "phase16_test_2", "type": "technical"},
            {"tweet_id": "phase16_test_3", "type": "other"},
        ],
        upsert_batch_size=2,
        embed_batch_size=2,
    )
    assert upserted == 3, "Upsert count mismatch."

    results = vector_store.query("where is my money", top_k=2)
    assert len(results) > 0, "No query results returned."
    assert {"id", "text", "metadata", "distance"}.issubset(results[0].keys())
    print("   VectorStore OK")

    print("3) Testing RagRetriever...")
    retriever = RagRetriever(vector_store=vector_store)
    retrieved = retriever.retrieve("refund issue", top_k=2)
    assert len(retrieved) > 0, "Retriever returned no results."
    print("   RagRetriever OK")

    print("\n✅ Phase 16 services smoke test passed.")


if __name__ == "__main__":
    run_phase16_smoke_test()

