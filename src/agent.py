"""
agent.py -- Main orchestrator for the SpotifyCares AI Support Agent.

Chains: preprocessing -> classification -> retrieval -> generation -> escalation
into a single process_message() call.

Includes API call caching for reproducible evaluation.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402
from src.data_pipeline.text_cleaner import clean_text  # noqa: E402
from src.intent.classifier import classify_message  # noqa: E402
from src.retrieval.retriever import retrieve, retrieval_evidence_strength, get_collection  # noqa: E402
from src.generation.reply_generator import generate_reply  # noqa: E402
from src.escalation.router import decide_escalation  # noqa: E402


class SpotifyCaresAgent:
    """
    AI Support Agent for SpotifyCares.

    Processes customer tweets through a full pipeline:
      1. Text preprocessing
      2. Hierarchical classification (route / domain / risk)
      3. Thread-aware RAG retrieval (training data only)
      4. Reply generation with safety checks
      5. Escalation decision per written policy
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
        retrieval_k: int | None = None,
    ):
        self.use_cache = use_cache
        self.retrieval_k = retrieval_k or config.RETRIEVAL_K
        self.cache_dir = Path(cache_dir) if cache_dir else config.RESULTS_DIR / "cached_outputs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-load ChromaDB collection
        self._collection = None

    @property
    def collection(self):
        """Lazy-load the ChromaDB collection if available."""
        if self._collection is None:
            try:
                self._collection = get_collection()
            except Exception:
                self._collection = None
        return self._collection

    def _cache_key(
        self,
        text: str,
        thread_id: str | None = None,
        thread_context: str = "",
        model: str = "",
        provider: str = "",
        k: int | None = None,
        version: str = "v2_rag_grounded",
    ) -> str:
        """Generate a deterministic multi-factor cache key."""
        payload = f"{version}|{provider}|{model}|{k}|{thread_id or 'none'}|{thread_context}|{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_cached(self, key: str) -> dict | None:
        """Load cached result if available."""
        if not self.use_cache:
            return None
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_cache(self, key: str, result: dict):
        """Save result to cache."""
        if not self.use_cache:
            return
        cache_path = self.cache_dir / f"{key}.json"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def process_message(
        self,
        tweet_text: str,
        thread_id: str | None = None,
        thread_context: str = "",
        thread_length: int = 1,
        force_refresh: bool = False,
    ) -> dict:
        """
        Process a customer tweet through the full agent pipeline.

        Args:
            tweet_text: Raw customer tweet text.
            thread_id: Thread ID for leakage prevention in retrieval.
            thread_context: Earlier messages in the thread (for classification context).
            thread_length: Number of messages in the thread.
            force_refresh: Whether to bypass and overwrite the cache.

        Returns:
            dict with keys: input, classification, retrieval, generation, escalation
        """
        # Check cache
        cache_key = self._cache_key(
            text=tweet_text,
            thread_id=thread_id,
            thread_context=thread_context,
            model=config.LLM_MODEL,
            provider=config.LLM_PROVIDER,
            k=self.retrieval_k,
        )
        if not force_refresh:
            cached = self._load_cached(cache_key)
            if cached:
                return cached

        # 1. Preprocess
        cleaned = clean_text(tweet_text)

        # 2. Classify
        classification = classify_message(cleaned, thread_context=thread_context)

        # 3. Retrieve (with thread exclusion for leakage prevention)
        retrieved = []
        evidence = "weak"
        try:
            retrieved = retrieve(
                cleaned,
                exclude_thread_id=thread_id,
                domain_filter=classification.get("domain"),
                k=self.retrieval_k,
                collection=self.collection,
            )
            evidence = retrieval_evidence_strength(retrieved)
        except Exception as e:
            print(f"[agent] Retrieval error: {e}")

        # 4. Generate reply
        gen_result = generate_reply(
            customer_message=cleaned,
            classification=classification,
            retrieved=retrieved,
            evidence_strength=evidence,
        )

        # 5. Escalation decision
        escalation = decide_escalation(
            classification=classification,
            evidence_strength=evidence,
            thread_length=thread_length,
        )

        # Assemble result
        result = {
            "input": {
                "raw_text": tweet_text,
                "clean_text": cleaned,
                "thread_id": thread_id,
                "thread_context": thread_context,
                "thread_length": thread_length,
            },
            "classification": classification,
            "retrieval": {
                "num_results": len(retrieved),
                "evidence_strength": evidence,
                "k": self.retrieval_k,
                "top_results": [
                    {
                        "customer_msg": r["customer_msg"][:150],
                        "agent_response": r["agent_response"][:200],
                        "similarity_score": r["similarity_score"],
                    }
                    for r in retrieved[:3]
                ],
            },
            "generation": gen_result,
            "escalation": escalation,
        }

        # Cache
        self._save_cache(cache_key, result)

        return result


def process_batch(
    messages: list[dict],
    agent: SpotifyCaresAgent | None = None,
) -> list[dict]:
    """
    Process a batch of messages.

    Each message dict should have: text, thread_id (optional), thread_context (optional), thread_length (optional).
    """
    if agent is None:
        agent = SpotifyCaresAgent()

    results = []
    for i, msg in enumerate(messages):
        result = agent.process_message(
            tweet_text=msg["text"],
            thread_id=msg.get("thread_id"),
            thread_context=msg.get("thread_context", ""),
            thread_length=msg.get("thread_length", 1),
        )
        results.append(result)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(messages)}")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = SpotifyCaresAgent(use_cache=False)

    test_messages = [
        "My Spotify keeps crashing every time I open it on my iPhone",
        "Someone hacked my account and changed my email and password",
        "I was charged $9.99 but I cancelled my subscription last week",
        "Thanks for fixing it! All sorted now",
        "How do I make a collaborative playlist?",
    ]

    for msg in test_messages:
        print(f"\n{'='*60}")
        print(f"Customer: {msg}")
        result = agent.process_message(msg)
        print(f"Route: {result['classification']['route']}")
        print(f"Domain: {result['classification']['domain']}")
        print(f"Risk: {result['classification']['risk_flag']}")
        print(f"Escalation: {result['escalation']['decision']} -- {result['escalation']['reason']}")
        print(f"Reply: {result['generation']['reply']}")
