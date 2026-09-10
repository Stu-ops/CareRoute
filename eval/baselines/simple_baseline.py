"""
simple_baseline.py -- TF-IDF + Logistic Regression + 1-NN retrieval + rule-based escalation.

A credible non-LLM baseline trained ONLY on training_labels.jsonl.
Never touches dev or test data during training.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402


class SimpleBaseline:
    """
    TF-IDF + Logistic Regression for classification,
    1-NN retrieval for reply generation,
    rule-based keyword matching for escalation.
    """

    def __init__(self, training_labels_path: str | Path | None = None):
        self.classifier_route = None
        self.classifier_domain = None
        self.vectorizer_route = None
        self.vectorizer_domain = None
        self._trained = False

        # 1-NN retrieval index
        self._retrieval_texts = []
        self._retrieval_responses = []
        self._retrieval_vectorizer = None
        self._retrieval_matrix = None

        if training_labels_path and Path(training_labels_path).exists():
            self._train(training_labels_path)

    def _train(self, labels_path: str | Path):
        """Train classifiers from labeled training data."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            print("[simple_baseline] scikit-learn not installed. Run: pip install scikit-learn")
            return

        texts, routes, domains = [], [], []
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                text = rec.get("clean_text", rec.get("text", ""))
                texts.append(text)
                routes.append(rec.get("route", "support"))
                domains.append(rec.get("domain", "unknown") or "unknown")

        if len(texts) < 10:
            print(f"[simple_baseline] Only {len(texts)} training examples -- too few")
            return

        # Route classifier
        self.vectorizer_route = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_route = self.vectorizer_route.fit_transform(texts)
        self.classifier_route = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
        self.classifier_route.fit(X_route, routes)

        # Domain classifier (only on support examples)
        support_idx = [i for i, r in enumerate(routes) if r == "support"]
        if len(support_idx) >= 5:
            support_texts = [texts[i] for i in support_idx]
            support_domains = [domains[i] for i in support_idx]
            self.vectorizer_domain = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X_domain = self.vectorizer_domain.fit_transform(support_texts)
            self.classifier_domain = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
            self.classifier_domain.fit(X_domain, support_domains)

        self._trained = True
        print(f"[simple_baseline] Trained on {len(texts)} examples")

    def load_retrieval_index(self, threads_path: str | Path = config.SPOTIFY_THREADS_JSONL):
        """
        Build 1-NN retrieval index from training-split threads.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            return

        manifest_path = config.SPLIT_MANIFEST_JSON
        if not Path(manifest_path).exists():
            return

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        texts, responses = [], []
        with open(threads_path, "r", encoding="utf-8") as f:
            for line in f:
                thread = json.loads(line)
                if manifest.get(thread["thread_id"]) != "train":
                    continue
                messages = thread["messages"]
                for i, msg in enumerate(messages):
                    if msg["inbound"]:
                        for j in range(i + 1, len(messages)):
                            if not messages[j]["inbound"]:
                                texts.append(msg.get("clean_text", msg["text"]))
                                responses.append(messages[j].get("clean_text", messages[j]["text"]))
                                break

        if texts:
            self._retrieval_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
            self._retrieval_matrix = self._retrieval_vectorizer.fit_transform(texts)
            self._retrieval_texts = texts
            self._retrieval_responses = responses
            print(f"[simple_baseline] Built retrieval index: {len(texts)} pairs")

    def _classify(self, text: str) -> dict:
        """Classify using TF-IDF + LR."""
        if not self._trained:
            return {
                "route": "support",
                "domain": None,
                "risk_flag": "none",
                "confidence": 0.5,
                "reasoning": "Simple baseline: not trained",
            }

        # Route
        X = self.vectorizer_route.transform([text])
        route = self.classifier_route.predict(X)[0]
        route_proba = max(self.classifier_route.predict_proba(X)[0])

        # Domain
        domain = None
        domain_proba = 0.0
        if route == "support" and self.classifier_domain is not None:
            X_d = self.vectorizer_domain.transform([text])
            domain = self.classifier_domain.predict(X_d)[0]
            domain_proba = max(self.classifier_domain.predict_proba(X_d)[0])

        # Risk flag (keyword-based)
        risk_flag = self._detect_risk(text)

        confidence = route_proba * (domain_proba if domain else 1.0)

        return {
            "route": route,
            "domain": domain if domain != "unknown" else None,
            "risk_flag": risk_flag,
            "confidence": round(confidence, 4),
            "reasoning": "Simple baseline: TF-IDF + Logistic Regression + keyword risk",
        }

    def _detect_risk(self, text: str) -> str:
        """Keyword-based risk flag detection."""
        lower = text.lower()
        if any(w in lower for w in ["hack", "unauthorized", "stolen", "someone changed"]):
            return "security"
        if any(w in lower for w in ["lawyer", "lawsuit", "sue", "legal", "gdpr"]):
            return "legal"
        if any(w in lower for w in ["charged twice", "unauthorized charge", "dispute", "refund"]):
            return "payment_dispute"
        if any(w in lower for w in ["again", "third time", "still not", "keep asking", "multiple times"]):
            return "repeated_contact"
        return "none"

    def _retrieve_1nn(self, text: str) -> str:
        """1-NN retrieval: return the most similar historical response."""
        if not self._retrieval_matrix is not None or not self._retrieval_responses:
            return "Hi there! We'd love to help. Could you send us a DM with more details? /AI"

        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            return "Hi there! We'd love to help. Could you send us a DM with more details? /AI"

        query_vec = self._retrieval_vectorizer.transform([text])
        similarities = cosine_similarity(query_vec, self._retrieval_matrix)[0]
        best_idx = similarities.argmax()
        return self._retrieval_responses[best_idx]

    def _escalation_rules(self, text: str, risk_flag: str, thread_length: int = 1) -> dict:
        """Transparent rule-based escalation."""
        lower = text.lower()

        if risk_flag in ("security", "legal"):
            return {
                "decision": "escalate",
                "confidence": 1.0,
                "reason": f"Keyword match: risk={risk_flag}",
                "policy_rule_matched": f"Simple baseline: {risk_flag} keyword -> escalate",
            }
        if risk_flag == "payment_dispute":
            return {
                "decision": "escalate",
                "confidence": 0.8,
                "reason": "Keyword match: billing dispute",
                "policy_rule_matched": "Simple baseline: payment keyword -> escalate",
            }
        if thread_length >= 5:
            return {
                "decision": "escalate",
                "confidence": 0.7,
                "reason": f"Thread length {thread_length} >= 5",
                "policy_rule_matched": "Simple baseline: long thread -> escalate",
            }
        return {
            "decision": "auto_handle",
            "confidence": 0.6,
            "reason": "No escalation triggers matched",
            "policy_rule_matched": "Simple baseline: default auto_handle",
        }

    def process_message(self, tweet_text: str, thread_length: int = 1, **kwargs) -> dict:
        """Process a message with the simple baseline pipeline."""
        from src.data_pipeline.text_cleaner import clean_text

        cleaned = clean_text(tweet_text)
        classification = self._classify(cleaned)
        reply = self._retrieve_1nn(cleaned)
        escalation = self._escalation_rules(cleaned, classification["risk_flag"], thread_length)

        return {
            "input": {"raw_text": tweet_text, "clean_text": cleaned},
            "classification": classification,
            "retrieval": {"num_results": 1, "evidence_strength": "moderate", "k": 1, "top_results": []},
            "generation": {
                "reply": reply,
                "safety_check": {"passed": True, "violations": []},
                "fallback_used": False,
                "raw_reply": reply,
                "reason": "Simple baseline: 1-NN retrieval (most similar historical response)",
            },
            "escalation": escalation,
        }


if __name__ == "__main__":
    baseline = SimpleBaseline()
    result = baseline.process_message("My Spotify keeps crashing on iPhone")
    print(json.dumps(result, indent=2))
