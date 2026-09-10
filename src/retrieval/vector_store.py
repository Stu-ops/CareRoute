"""
vector_store.py -- ChromaDB-based vector store for historical conversations.

CRITICAL RULES:
  1. Only indexes training split threads (dev/test excluded at index time).
  2. Stores {customer_msg, agent_response, thread_context} triples.
  3. Metadata includes thread_id for thread-exclusion during retrieval.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402


def _load_manifest() -> dict[str, str]:
    """Load the split manifest (thread_id -> split)."""
    with open(config.SPLIT_MANIFEST_JSON, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_training_pairs(
    threads_path: Path = config.SPOTIFY_THREADS_JSONL,
) -> list[dict]:
    """
    Extract (customer_message, agent_response, thread_context) triples
    from training-split threads only.
    """
    manifest = _load_manifest()
    pairs = []

    with open(threads_path, "r", encoding="utf-8") as fh:
        for line in fh:
            thread = json.loads(line)
            thread_id = thread["thread_id"]

            # Only training split
            if manifest.get(thread_id) != "train":
                continue

            messages = thread["messages"]
            for i, msg in enumerate(messages):
                # Find inbound customer messages followed by outbound brand responses
                if msg["inbound"]:
                    # Look for the next outbound response
                    for j in range(i + 1, len(messages)):
                        if not messages[j]["inbound"]:
                            # Build thread context (all messages before this one)
                            context = " | ".join(
                                m.get("clean_text", m["text"])
                                for m in messages[:i]
                            )
                            pairs.append({
                                "customer_msg": msg.get("clean_text", msg["text"]),
                                "agent_response": messages[j].get(
                                    "clean_text", messages[j]["text"]
                                ),
                                "thread_id": thread_id,
                                "thread_context": context,
                                "customer_tweet_id": msg["tweet_id"],
                                "agent_tweet_id": messages[j]["tweet_id"],
                            })
                            break  # only pair with first response

    return pairs


def build_vector_store(
    threads_path: str | Path = config.SPOTIFY_THREADS_JSONL,
    db_dir: str | Path = config.CHROMA_DB_DIR,
    collection_name: str = "spotify_support",
) -> dict:
    """
    Build the ChromaDB vector store from training-split conversation pairs.

    Uses sentence-transformers for embeddings (free, reproducible).
    """
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("[vector_store] chromadb not installed. Run: pip install chromadb")
        return {"error": "chromadb not installed"}

    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    # Load training pairs
    pairs = _load_training_pairs(Path(threads_path))
    print(f"[vector_store] Loaded {len(pairs):,} training conversation pairs")

    if not pairs:
        return {"error": "No training pairs found"}

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=str(db_dir))

    # Use sentence-transformers embedding function
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )

    # Delete existing collection if it exists (rebuild)
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"description": "SpotifyCares training conversation pairs"},
    )

    # Batch insert (ChromaDB has a limit of ~5000 per batch)
    batch_size = 4000
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        collection.add(
            ids=[f"pair_{start + i}" for i in range(len(batch))],
            documents=[p["customer_msg"] for p in batch],
            metadatas=[
                {
                    "agent_response": p["agent_response"],
                    "thread_id": p["thread_id"],
                    "thread_context": p["thread_context"][:500],  # truncate for storage
                    "customer_tweet_id": p["customer_tweet_id"],
                    "agent_tweet_id": p["agent_tweet_id"],
                }
                for p in batch
            ],
        )
        print(f"  Indexed batch {start // batch_size + 1} ({len(batch)} pairs)")

    summary = {
        "total_pairs_indexed": len(pairs),
        "collection_name": collection_name,
        "db_dir": str(db_dir),
        "embedding_model": config.EMBEDDING_MODEL,
    }
    print(f"[vector_store] Done -- {len(pairs):,} pairs indexed")
    return summary


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = build_vector_store()
    print(json.dumps(result, indent=2))
