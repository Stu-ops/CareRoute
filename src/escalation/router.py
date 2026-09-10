"""
router.py -- Escalation decision engine.

Implements the written escalation policy (docs/escalation_policy.md).
Two-stage decision:
  1. Hard rules from risk flags -> deterministic escalation for security/legal/billing.
  2. Soft rules from confidence + evidence -> coverage-risk calibrated threshold.

Returns: {decision, confidence, reason, policy_rule_matched}
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402


# ── Hard escalation rules (from escalation_policy.md) ────────────────────────
HARD_ESCALATE_RULES = {
    "security": {
        "decision": "escalate",
        "reason": "Account security concern -- must move to private support",
        "policy_rule": "Risk: security -> always escalate",
    },
    "legal": {
        "decision": "escalate",
        "reason": "Legal language detected -- must involve internal team",
        "policy_rule": "Risk: legal -> always escalate",
    },
    "payment_dispute": {
        "decision": "escalate",
        "reason": "Payment dispute -- agent cannot verify charges or issue refunds",
        "policy_rule": "Risk: payment_dispute -> escalate to human",
    },
}

# ── Soft escalation rules ────────────────────────────────────────────────────
SOFT_ESCALATE_RULES = {
    "repeated_contact": {
        "decision": "escalate",
        "reason": "Customer has contacted multiple times without resolution",
        "policy_rule": "Risk: repeated_contact -> escalate to human",
    },
    "unclear": {
        "decision": "clarify",
        "reason": "Intent is unclear -- ask clarifying question before proceeding",
        "policy_rule": "Risk: unclear -> clarify first",
    },
}

# ── Route-based rules ────────────────────────────────────────────────────────
ROUTE_RULES = {
    "feedback": {
        "decision": "auto_handle",
        "reason": "Positive/neutral feedback -- acknowledge only",
        "policy_rule": "Route: feedback -> acknowledge (auto)",
    },
    "abuse_spam": {
        "decision": "auto_handle",
        "reason": "Abuse/spam -- ignore or flag",
        "policy_rule": "Route: abuse_spam -> ignore/flag",
    },
}


def decide_escalation(
    classification: dict,
    evidence_strength: str = "moderate",
    thread_length: int = 1,
    confidence_threshold: float | None = None,
) -> dict:
    """
    Apply the escalation policy to determine whether to auto-handle, escalate, or clarify.

    Args:
        classification: Dict with route, domain, risk_flag, confidence.
        evidence_strength: "strong", "moderate", or "weak" from retriever.
        thread_length: Number of messages in the thread (for repeated_contact detection).
        confidence_threshold: Override confidence threshold (default from config).

    Returns:
        dict with: decision, confidence, reason, policy_rule_matched
    """
    route = classification.get("route", "support")
    domain = classification.get("domain")
    risk_flag = classification.get("risk_flag", "none")
    classifier_confidence = classification.get("confidence", 0.5)
    threshold = confidence_threshold or config.ESCALATION_CONFIDENCE_THRESHOLD

    # 1. Route-based rules (feedback, abuse/spam)
    if route in ROUTE_RULES:
        rule = ROUTE_RULES[route]
        return {
            "decision": rule["decision"],
            "confidence": classifier_confidence,
            "reason": rule["reason"],
            "policy_rule_matched": rule["policy_rule"],
        }

    # 2. Hard escalation rules (security, legal, payment_dispute)
    if risk_flag in HARD_ESCALATE_RULES:
        rule = HARD_ESCALATE_RULES[risk_flag]
        return {
            "decision": rule["decision"],
            "confidence": classifier_confidence,
            "reason": rule["reason"],
            "policy_rule_matched": rule["policy_rule"],
        }

    # 3. Thread-length-based repeated_contact detection
    if thread_length >= 5 or risk_flag == "repeated_contact":
        rule = SOFT_ESCALATE_RULES["repeated_contact"]
        return {
            "decision": rule["decision"],
            "confidence": classifier_confidence,
            "reason": rule["reason"],
            "policy_rule_matched": rule["policy_rule"],
        }

    # 4. Unclear intent
    if risk_flag == "unclear":
        rule = SOFT_ESCALATE_RULES["unclear"]
        return {
            "decision": rule["decision"],
            "confidence": classifier_confidence,
            "reason": rule["reason"],
            "policy_rule_matched": rule["policy_rule"],
        }

    # 5. Confidence-based soft escalation
    if classifier_confidence < threshold:
        return {
            "decision": "clarify",
            "confidence": classifier_confidence,
            "reason": (
                f"Classifier confidence ({classifier_confidence:.2f}) "
                f"below threshold ({threshold:.2f})"
            ),
            "policy_rule_matched": "Low calibrated confidence -> clarify/escalate",
        }

    # 6. Evidence-strength-based fallback
    if evidence_strength == "weak":
        return {
            "decision": "clarify",
            "confidence": classifier_confidence,
            "reason": "Weak retrieval evidence -- insufficient grounding for auto-response",
            "policy_rule_matched": "Weak evidence -> DM redirect / clarify",
        }

    # 7. Auto-handle (all checks passed)
    return {
        "decision": "auto_handle",
        "confidence": classifier_confidence,
        "reason": f"Standard {domain or 'support'} issue with sufficient evidence",
        "policy_rule_matched": f"Domain: {domain} + risk: none + evidence: {evidence_strength} -> auto_handle",
    }


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    test_cases = [
        ({"route": "support", "domain": "playback", "risk_flag": "none", "confidence": 0.85}, "strong", 2),
        ({"route": "support", "domain": "account", "risk_flag": "security", "confidence": 0.9}, "strong", 1),
        ({"route": "support", "domain": "billing", "risk_flag": "payment_dispute", "confidence": 0.8}, "moderate", 1),
        ({"route": "feedback", "domain": None, "risk_flag": "none", "confidence": 0.95}, "strong", 1),
        ({"route": "support", "domain": "app_device", "risk_flag": "none", "confidence": 0.3}, "moderate", 1),
        ({"route": "support", "domain": "playback", "risk_flag": "none", "confidence": 0.7}, "weak", 1),
        ({"route": "support", "domain": "account", "risk_flag": "repeated_contact", "confidence": 0.8}, "moderate", 7),
    ]
    for cls, evidence, thread_len in test_cases:
        result = decide_escalation(cls, evidence, thread_len)
        print(f"\n{cls['domain'] or cls['route']} (risk={cls['risk_flag']}, conf={cls['confidence']}, evidence={evidence}, thread={thread_len})")
        print(f"  -> {result['decision']}: {result['reason']}")
