"""
trivial_baseline.py -- Most-frequent-class + template response baseline.

This is the floor. Any real system must beat this convincingly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402

# Template response (matches SpotifyCares DM-redirect pattern, 30.8% of real responses)
TEMPLATE_REPLY = (
    "Hi there! We'd love to help with that. "
    "Could you send us a DM with more details? /AI"
)

# Most-frequent class (will be determined from training data)
DEFAULT_ROUTE = "support"
DEFAULT_DOMAIN = "playback"
DEFAULT_RISK = "none"


class TrivialBaseline:
    """Always predicts the most-frequent class and returns a template reply."""

    def __init__(self, training_labels_path: str | Path | None = None):
        self.most_freq_route = DEFAULT_ROUTE
        self.most_freq_domain = DEFAULT_DOMAIN
        self.most_freq_risk = DEFAULT_RISK

        # If training labels exist, compute actual most-frequent classes
        if training_labels_path and Path(training_labels_path).exists():
            self._compute_frequencies(training_labels_path)

    def _compute_frequencies(self, path: str | Path):
        """Compute most-frequent class from training labels."""
        from collections import Counter

        routes, domains, risks = Counter(), Counter(), Counter()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                routes[rec.get("route", "support")] += 1
                if rec.get("domain"):
                    domains[rec["domain"]] += 1
                risks[rec.get("risk_flag", "none")] += 1

        if routes:
            self.most_freq_route = routes.most_common(1)[0][0]
        if domains:
            self.most_freq_domain = domains.most_common(1)[0][0]
        if risks:
            self.most_freq_risk = risks.most_common(1)[0][0]

    def process_message(self, tweet_text: str, **kwargs) -> dict:
        """Process a message with the trivial baseline."""
        return {
            "input": {"raw_text": tweet_text, "clean_text": tweet_text},
            "classification": {
                "route": self.most_freq_route,
                "domain": self.most_freq_domain if self.most_freq_route == "support" else None,
                "risk_flag": self.most_freq_risk,
                "confidence": 1.0,
                "reasoning": "Trivial baseline: always predicts most-frequent class",
            },
            "retrieval": {"num_results": 0, "evidence_strength": "none", "k": 0, "top_results": []},
            "generation": {
                "reply": TEMPLATE_REPLY,
                "safety_check": {"passed": True, "violations": []},
                "fallback_used": False,
                "raw_reply": TEMPLATE_REPLY,
                "reason": "Trivial baseline: template response",
            },
            "escalation": {
                "decision": "auto_handle",  # never-escalate variant
                "confidence": 1.0,
                "reason": "Trivial baseline: never escalate",
                "policy_rule_matched": "N/A (trivial baseline)",
            },
        }


# ── Variant: always-escalate ─────────────────────────────────────────────────
class TrivialBaselineAlwaysEscalate(TrivialBaseline):
    """Same as TrivialBaseline but always escalates."""

    def process_message(self, tweet_text: str, **kwargs) -> dict:
        result = super().process_message(tweet_text, **kwargs)
        result["escalation"] = {
            "decision": "escalate",
            "confidence": 1.0,
            "reason": "Trivial baseline: always escalate",
            "policy_rule_matched": "N/A (trivial baseline)",
        }
        return result


if __name__ == "__main__":
    baseline = TrivialBaseline()
    result = baseline.process_message("My Spotify keeps crashing")
    print(json.dumps(result, indent=2))
