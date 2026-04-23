"""
Ingest conversations_for_rag.csv into Chroma as conversation-level documents.

Run from backend directory:
    uv run python scripts/ingest_conversations.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

CSV_CHUNK_SIZE = 250_000
UPSERT_BATCH_SIZE = 256
EMBED_BATCH_SIZE = 64
COLLECTION_NAME = "support_conversations"


def _project_root() -> Path:
    # scripts/ -> backend/ -> project_root/
    return Path(__file__).resolve().parents[2]


def _resolve_input_csv(project_root: Path) -> Path:
    candidates = [
        project_root / "data" / "cleaned" / "conversations_for_rag.csv",
        project_root / "backend" / "data" / "cleaned" / "conversations_for_rag.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing input CSV. Checked:\n{checked}")


def _normalize_inbound(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _aggregate_chunk_to_conversations(chunk: pd.DataFrame) -> list[dict[str, Any]]:
    chunk = chunk.copy()
    chunk["text_for_rag"] = chunk["text_for_rag"].fillna("").astype(str).str.strip()
    chunk = chunk[chunk["text_for_rag"] != ""]
    chunk = chunk.dropna(subset=["conversation_id", "tweet_id"])
    if chunk.empty:
        return []

    chunk["tweet_id"] = chunk["tweet_id"].astype(str)
    chunk["conversation_id"] = chunk["conversation_id"].astype(str)
    chunk["position"] = pd.to_numeric(chunk["position"], errors="coerce")
    chunk = chunk.sort_values(["conversation_id", "position", "tweet_id"], kind="mergesort")

    grouped_records: list[dict[str, Any]] = []
    for conversation_id, grp in chunk.groupby("conversation_id", sort=False):
        ordered_text = " ".join(grp["text_for_rag"].tolist()).strip()
        if not ordered_text:
            continue

        grouped_records.append(
            {
                "id": conversation_id,
                "text": ordered_text,
                "metadata": {
                    "conversation_id": conversation_id,
                    "tweet_count": int(len(grp)),
                    "first_tweet_id": str(grp["tweet_id"].iloc[0]),
                    "last_tweet_id": str(grp["tweet_id"].iloc[-1]),
                    "first_author_id": str(grp["author_id"].iloc[0]) if "author_id" in grp.columns else "",
                    "inbound_any": bool(grp["inbound"].map(_normalize_inbound).any()) if "inbound" in grp.columns else False,
                },
            }
        )
    return grouped_records


def ingest_full_dataset() -> int:
    project_root = _project_root()
    csv_path = _resolve_input_csv(project_root)
    persist_dir = project_root / "data" / "chroma_db"

    vector_store = VectorStore(
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
    )

    required_cols = {"tweet_id", "conversation_id", "position", "text_for_rag", "author_id", "inbound"}
    indexed_total = 0
    seen_conversation_ids: set[str] = set()

    for chunk_idx, chunk in enumerate(
        pd.read_csv(csv_path, chunksize=CSV_CHUNK_SIZE, low_memory=False),
        start=1,
    ):
        missing = required_cols - set(chunk.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        records = _aggregate_chunk_to_conversations(chunk)
        # De-duplicate across chunks (in case same conversation spills chunks).
        records = [r for r in records if r["id"] not in seen_conversation_ids]
        for record in records:
            seen_conversation_ids.add(record["id"])

        if not records:
            logger.info("Chunk %d: no valid conversation records.", chunk_idx)
            continue

        ids = [r["id"] for r in records]
        texts = [r["text"] for r in records]
        metadatas = [r["metadata"] for r in records]

        indexed = vector_store.upsert_texts(
            ids=ids,
            texts=texts,
            metadatas=metadatas,
            upsert_batch_size=UPSERT_BATCH_SIZE,
            embed_batch_size=EMBED_BATCH_SIZE,
        )
        indexed_total += indexed
        logger.info(
            "Chunk %d: indexed=%d total_indexed=%d collection_count=%d",
            chunk_idx,
            indexed,
            indexed_total,
            vector_store.count(),
        )

    logger.info(
        "Ingestion complete. indexed_total=%d collection_count=%d persist_dir=%s",
        indexed_total,
        vector_store.count(),
        persist_dir,
    )
    return indexed_total


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    ingest_full_dataset()


if __name__ == "__main__":
    main()

