"""
Build Chroma vector index from conversations_for_rag.csv.

Usage example:
    uv run python -m app.scripts.build_vector_index
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _to_py_scalar(value: Any) -> Any:
    """Convert pandas/numpy scalars to plain Python types."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover
            return value
    return value


def _build_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "tweet_id": str(_to_py_scalar(record["tweet_id"])),
        "conversation_id": str(_to_py_scalar(record["conversation_id"])),
        "position": int(_to_py_scalar(record["position"])) if record.get("position") is not None else -1,
        "author_id": str(_to_py_scalar(record["author_id"])) if record.get("author_id") is not None else "",
        "inbound": bool(_to_py_scalar(record["inbound"])) if record.get("inbound") is not None else False,
    }


def build_index(
    csv_path: Path,
    *,
    collection_name: str,
    persist_directory: Path,
    csv_chunk_size: int,
    upsert_batch_size: int,
    embed_batch_size: int,
    max_rows: int | None = None,
) -> int:
    """Build or update vector index from CSV input."""
    required_cols = {
        "tweet_id",
        "conversation_id",
        "position",
        "text_for_rag",
        "author_id",
        "inbound",
    }

    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    vector_store = VectorStore(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )

    processed = 0
    indexed = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(csv_path, chunksize=csv_chunk_size, low_memory=False),
        start=1,
    ):
        missing = required_cols - set(chunk.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if max_rows is not None and processed >= max_rows:
            break

        chunk = chunk.copy()
        chunk["text_for_rag"] = chunk["text_for_rag"].fillna("").astype(str).str.strip()
        chunk = chunk[chunk["text_for_rag"] != ""]
        chunk = chunk.dropna(subset=["tweet_id"])
        chunk["tweet_id"] = chunk["tweet_id"].astype(str)
        chunk = chunk.drop_duplicates(subset=["tweet_id"], keep="last")

        if max_rows is not None:
            remaining = max_rows - processed
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.head(remaining)

        records = chunk.to_dict(orient="records")
        ids = [str(rec["tweet_id"]) for rec in records]
        texts = [str(rec["text_for_rag"]) for rec in records]
        metadatas = [_build_metadata(rec) for rec in records]

        if ids:
            upserted = vector_store.upsert_texts(
                ids=ids,
                texts=texts,
                metadatas=metadatas,
                upsert_batch_size=upsert_batch_size,
                embed_batch_size=embed_batch_size,
            )
            indexed += upserted

        processed += len(records)
        logger.info(
            "Chunk %d done: processed=%d indexed=%d collection_count=%d",
            chunk_idx,
            processed,
            indexed,
            vector_store.count(),
        )

    logger.info(
        "Index build complete: indexed=%d collection_count=%d path=%s",
        indexed,
        vector_store.count(),
        persist_directory,
    )
    return indexed


def parse_args() -> argparse.Namespace:
    project_root = _project_root()
    parser = argparse.ArgumentParser(description="Build Chroma vector index for RAG.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=project_root / "data" / "cleaned" / "conversations_for_rag.csv",
        help="Path to conversations_for_rag CSV file.",
    )
    parser.add_argument(
        "--persist-directory",
        type=Path,
        default=project_root / "data" / "chroma_db",
        help="Persistent Chroma directory.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=VectorStore.DEFAULT_COLLECTION_NAME,
        help="Chroma collection name.",
    )
    parser.add_argument("--csv-chunk-size", type=int, default=25_000, help="CSV chunk size.")
    parser.add_argument("--upsert-batch-size", type=int, default=512, help="Vector upsert batch size.")
    parser.add_argument("--embed-batch-size", type=int, default=64, help="Embedding model batch size.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    build_index(
        csv_path=args.csv_path,
        collection_name=args.collection_name,
        persist_directory=args.persist_directory,
        csv_chunk_size=args.csv_chunk_size,
        upsert_batch_size=args.upsert_batch_size,
        embed_batch_size=args.embed_batch_size,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()

