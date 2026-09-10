"""
classifier.py -- Hierarchical intent classification (Route -> Domain -> Risk Flag).

Supports two modes:
  1. LLM-based (default) -- uses the configured LLM via API for zero/few-shot classification.
  2. Embedding-based (fallback) -- uses sentence-transformers + cosine similarity.

Returns: {route, domain, risk_flag, confidence, reasoning}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402

# ── Taxonomy definitions ─────────────────────────────────────────────────────
ROUTES = ["support", "feedback", "abuse_spam"]
DOMAINS = ["playback", "account", "billing", "content", "app_device", "how_to"]
RISK_FLAGS = ["security", "payment_dispute", "legal", "repeated_contact", "unclear", "none"]

CLASSIFICATION_PROMPT = """You are a customer support message classifier for SpotifyCares (Spotify's Twitter support).

Classify the following customer message into three layers:

## Layer 1 -- Route
- support: Customer is reporting a problem, asking a question, or requesting help.
- feedback: Customer is providing feedback, praise, or commentary without requesting action.
- abuse_spam: Abusive, spam, or completely off-topic.

## Layer 2 -- Domain (only if route = "support")
- playback: Issues with playing music (skipping, pausing, buffering, no sound).
- account: Login, password, account recovery, profile changes.
- billing: Payment issues, subscription changes, refund requests.
- content: Missing songs/albums/podcasts, region-locked content.
- app_device: App crashes, freezes, device-specific issues, Bluetooth, offline mode.
- how_to: How-to questions, feature discovery, general usage help.

## Layer 3 -- Risk Flag (always assigned)
- security: Account compromise, unauthorized access, suspicious activity.
- payment_dispute: Disputed charges, double billing, unauthorized payment.
- legal: Mentions of lawyers, lawsuits, regulatory complaints, GDPR.
- repeated_contact: Customer indicates multiple unresolved contacts (or detected from thread context).
- unclear: Message is too vague or ambiguous to classify confidently.
- none: Standard interaction, no special risk.

Thread context (if available): {thread_context}

Customer message: {message}

Respond in JSON format ONLY:
{{
  "route": "<route>",
  "domain": "<domain or null>",
  "risk_flag": "<risk_flag>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>"
}}"""


def classify_message(
    message: str,
    thread_context: str = "",
    provider: str | None = None,
    model: str | None = None,
) -> dict:
    """
    Classify a customer message using the configured LLM.

    Args:
        message: The customer tweet text (cleaned).
        thread_context: Optional context from earlier messages in the thread.
        provider: Override LLM provider (default from config).
        model: Override LLM model (default from config).

    Returns:
        dict with keys: route, domain, risk_flag, confidence, reasoning
    """
    provider = provider or config.LLM_PROVIDER
    model = model or config.LLM_MODEL

    prompt = CLASSIFICATION_PROMPT.format(
        message=message,
        thread_context=thread_context or "No prior context available.",
    )

    if provider == "openai":
        return _classify_openai(prompt, model, raw_message=message, thread_context=thread_context)
    else:
        # Fallback: keyword-based classifier for testing without API
        return _classify_keyword_fallback(message, thread_context)


def _classify_openai(prompt: str, model: str, raw_message: str = "", thread_context: str = "") -> dict:
    """Call OpenAI API for classification."""
    try:
        from openai import OpenAI
    except ImportError:
        print("[classifier] openai package not installed, using keyword fallback")
        return _classify_keyword_fallback(raw_message or prompt, thread_context)

    api_key = config.LLM_API_KEY or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("[classifier] No API key set, using keyword fallback")
        return _classify_keyword_fallback(raw_message or prompt, thread_context)

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise classifier. Respond only in valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=config.LLM_TEMPERATURE_CLASSIFY,
        max_tokens=200,
    )

    raw = response.choices[0].message.content.strip()
    # Extract JSON from response (handle markdown code blocks)
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "route": "support",
            "domain": None,
            "risk_flag": "unclear",
            "confidence": 0.1,
            "reasoning": f"Failed to parse LLM response: {raw[:100]}",
        }

    # Validate fields
    result.setdefault("route", "support")
    result.setdefault("domain", None)
    result.setdefault("risk_flag", "none")
    result.setdefault("confidence", 0.5)
    result.setdefault("reasoning", "")

    if result["route"] not in ROUTES:
        result["route"] = "support"
    if result["domain"] and result["domain"] not in DOMAINS:
        result["domain"] = None
    if result["risk_flag"] not in RISK_FLAGS:
        result["risk_flag"] = "none"

    return result


def _classify_keyword_fallback(message: str, thread_context: str = "") -> dict:
    """
    Simple keyword-based classifier for testing without LLM API.
    Not intended for production -- only for pipeline testing.
    """
    msg_lower = message.lower()

    # Route
    abuse_words = {"fuck", "shit", "idiot", "stupid", "damn", "ass"}
    feedback_words = {"thanks", "thank you", "love", "great", "awesome", "amazing", "best"}

    route = "support"
    if any(w in msg_lower for w in abuse_words) and len(message.split()) < 5:
        route = "abuse_spam"
    elif any(w in msg_lower for w in feedback_words) and "?" not in message:
        route = "feedback"

    # Domain
    domain = None
    if route == "support":
        if any(w in msg_lower for w in ["play", "skip", "pause", "buffer", "sound", "song", "music", "shuffle"]):
            domain = "playback"
        elif any(w in msg_lower for w in ["login", "log in", "password", "account", "sign in"]):
            domain = "account"
        elif any(w in msg_lower for w in ["charge", "bill", "refund", "payment", "subscription", "premium", "cancel"]):
            domain = "billing"
        elif any(w in msg_lower for w in ["missing", "available", "region", "album", "podcast"]):
            domain = "content"
        elif any(w in msg_lower for w in ["crash", "freeze", "bug", "bluetooth", "device", "phone", "iphone", "android", "offline"]):
            domain = "app_device"
        elif any(w in msg_lower for w in ["how", "where", "what is", "can i", "feature"]):
            domain = "how_to"

    # Risk flag
    risk_flag = "none"
    if any(w in msg_lower for w in ["charged twice", "unauthorized charge", "double charge", "dispute", "refund"]):
        risk_flag = "payment_dispute"
    elif any(w in msg_lower for w in ["hack", "unauthorized access", "unauthorized login", "stolen", "suspicious",
                                     "someone changed", "someone is using", "someone logged", "compromised"]):
        risk_flag = "security"
    elif any(w in msg_lower for w in ["lawyer", "lawsuit", "sue", "legal", "gdpr"]):
        risk_flag = "legal"
    elif any(w in msg_lower for w in ["again", "third time", "still not", "keep asking"]):
        risk_flag = "repeated_contact"

    return {
        "route": route,
        "domain": domain,
        "risk_flag": risk_flag,
        "confidence": 0.5,  # keyword fallback is never high confidence
        "reasoning": "Keyword-based fallback classifier (no LLM API configured)",
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_msgs = [
        "My Spotify keeps crashing on my iPhone",
        "Someone hacked my account and changed the email",
        "I was charged twice this month, want a refund",
        "Thanks for the help, all sorted now!",
        "How do I make a collaborative playlist?",
    ]
    for msg in test_msgs:
        result = classify_message(msg)
        print(f"\n{msg}")
        print(f"  -> {json.dumps(result)}")
