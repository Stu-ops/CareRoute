"""
retriever.py -- Thread-aware similarity search with leakage prevention.

CRITICAL RULE: For every query, the retrieval system MUST exclude:
  (a) the queried tweet's own historical reply
  (b) every message in the queried tweet's thread
This is enforced via thread_id filtering at query time.

Dual-Engine Architecture:
  - Primary: ChromaDB vector store with dense SentenceTransformer embeddings.
  - Fallback: In-memory TF-IDF index on training-split conversations (activated
    automatically when ChromaDB is not installed or the collection is absent).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Robust sys.path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# In-memory TF-IDF fallback cache
_FALLBACK_INDEX = None


def get_collection(
    db_dir: str | Path = config.CHROMA_DB_DIR,
    collection_name: str = "spotify_support",
):
    """Get a handle to the existing ChromaDB collection."""
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=str(db_dir))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )
    return client.get_collection(name=collection_name, embedding_function=ef)


def _get_fallback_index() -> dict | None:
    """Build or return cached in-memory TF-IDF index from training-split threads."""
    global _FALLBACK_INDEX
    if _FALLBACK_INDEX is not None:
        return _FALLBACK_INDEX

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return None

    manifest_path = config.SPLIT_MANIFEST_JSON
    threads_path = config.SPOTIFY_THREADS_JSONL
    if not Path(manifest_path).exists() or not Path(threads_path).exists():
        return None

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records = []
    texts = []
    with open(threads_path, "r", encoding="utf-8") as f:
        for line in f:
            thread = json.loads(line)
            tid = thread["thread_id"]
            if manifest.get(tid) != "train":
                continue

            messages = thread["messages"]
            for i, msg in enumerate(messages):
                if msg.get("inbound"):
                    for j in range(i + 1, len(messages)):
                        if not messages[j].get("inbound"):
                            c_text = msg.get("clean_text", msg["text"])
                            a_text = messages[j].get("clean_text", messages[j]["text"])
                            ctx = " | ".join(
                                m.get("clean_text", m["text"]) for m in messages[:i]
                            )
                            texts.append(c_text)
                            records.append({
                                "customer_msg": c_text,
                                "agent_response": a_text,
                                "thread_id": tid,
                                "thread_context": ctx,
                                "customer_tweet_id": msg.get("tweet_id", ""),
                                "agent_tweet_id": messages[j].get("tweet_id", ""),
                            })
                            break

    if not texts:
        return None

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    _FALLBACK_INDEX = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "records": records,
    }
    return _FALLBACK_INDEX


def retrieve(
    query: str,
    exclude_thread_id: str | None = None,
    domain_filter: str | None = None,
    k: int | None = None,
    collection=None,
) -> list[dict]:
    """
    Retrieve top-K similar historical conversations for a query.

    Args:
        query: The customer tweet text (cleaned).
        exclude_thread_id: Thread ID to exclude from results (leakage prevention).
        domain_filter: Optional domain filter for more relevant results.
        k: Number of results to return (default from config).
        collection: Optional pre-loaded ChromaDB collection.

    Returns:
        List of dicts with keys:
          customer_msg, agent_response, thread_id, similarity_score, thread_context
    """
    k = k or config.RETRIEVAL_K

    if collection is None:
        try:
            collection = get_collection()
        except Exception:
            collection = None

    # Path 1: ChromaDB collection available
    if collection is not None:
        try:
            fetch_k = k * 3 if exclude_thread_id else k
            results = collection.query(
                query_texts=[query],
                n_results=min(fetch_k, collection.count()),
            )

            if results and results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]

                candidates = []
                for doc, meta, dist in zip(documents, metadatas, distances):
                    thread_id = meta.get("thread_id", "")
                    if exclude_thread_id and thread_id == exclude_thread_id:
                        continue
                    if domain_filter and meta.get("domain") and meta["domain"] != domain_filter:
                        continue

                    similarity = max(0.0, 1.0 - dist / 2.0)
                    candidates.append({
                        "customer_msg": doc,
                        "agent_response": meta.get("agent_response", ""),
                        "thread_id": thread_id,
                        "thread_context": meta.get("thread_context", ""),
                        "similarity_score": round(similarity, 4),
                        "customer_tweet_id": meta.get("customer_tweet_id", ""),
                        "agent_tweet_id": meta.get("agent_tweet_id", ""),
                    })
                    if len(candidates) >= k:
                        break
                return candidates
        except Exception:
            pass

    # Path 2: Fallback in-memory TF-IDF index (training split only)
    fb_index = _get_fallback_index()
    if fb_index is None:
        return []

    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        vec = fb_index["vectorizer"]
        q_vec = vec.transform([query])
        sims = cosine_similarity(q_vec, fb_index["matrix"])[0]

        top_indices = np.argsort(sims)[::-1]
        candidates = []

        for idx in top_indices:
            score = float(sims[idx])
            if score <= 0.02:
                break
            rec = fb_index["records"][idx]
            if exclude_thread_id and rec["thread_id"] == exclude_thread_id:
                continue

            candidates.append({
                "customer_msg": rec["customer_msg"],
                "agent_response": rec["agent_response"],
                "thread_id": rec["thread_id"],
                "thread_context": rec["thread_context"],
                "similarity_score": round(score, 4),
                "customer_tweet_id": rec["customer_tweet_id"],
                "agent_tweet_id": rec["agent_tweet_id"],
            })
            if len(candidates) >= k:
                break

        return candidates
    except Exception:
        return []


def retrieval_evidence_strength(results: list[dict]) -> str:
    """
    Assess the strength of retrieval evidence.

    Returns: "strong", "moderate", or "weak"
    """
    if not results:
        return "weak"

    top_score = results[0]["similarity_score"]
    avg_score = sum(r["similarity_score"] for r in results) / len(results)

    if top_score >= 0.70 and avg_score >= 0.50:
        return "strong"
    elif top_score >= 0.35:
        return "moderate"
    else:
        return "weak"


if __name__ == "__main__":
    test_queries = [
        "My Spotify keeps skipping songs on my Bluetooth speaker",
        "I was charged twice this month",
        "How do I make a collaborative playlist?",
    ]
    for q in test_queries:
        results = retrieve(q, k=3)
        print(f"\nQuery: {q}")
        print(f"Evidence strength: {retrieval_evidence_strength(results)}")
        for i, r in enumerate(results):
            print(f"  [{i+1}] sim={r['similarity_score']:.3f} | tid={r['thread_id']}")
            print(f"      Customer: {r['customer_msg'][:70]}...")
            print(f"      Agent:    {r['agent_response'][:70]}...")
