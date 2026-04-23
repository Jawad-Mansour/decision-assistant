"""
Validate retrieval quality from the default VectorStore collection.

Run:
    uv run python scripts/validate_vector_store.py
"""

from __future__ import annotations

from app.services.vector_store import VectorStore


def main() -> None:
    store = VectorStore.get_instance()
    print(f"Total conversations in store: {store.count():,}")

    queries = [
        "my refund is not showing up",
        "phone is broken",
        "cancel my order",
        "help with my bill",
        "internet is down",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        results = store.query(q, top_k=2)
        for r in results:
            print(f"  distance={r['distance']:.4f} | id={r['id']}")
            text = (r.get("text") or "")[:120]
            print(f"  text: {text}...")


if __name__ == "__main__":
    main()

