"""
reply_generator.py -- LLM-based reply drafting with safety checks.

Generates a customer support reply grounded in retrieved historical conversations.
Applies hard safety checks post-generation:
  1. Character limit (280 chars, Twitter)
  2. No invented URLs
  3. No PII requests in public
  4. No fabricated policies / refunds / account actions
  5. Fallback to DM redirect when evidence is weak
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402
from src.generation.voice_guidelines import get_voice_prompt_section  # noqa: E402

# ── Safety patterns ──────────────────────────────────────────────────────────
_URL_RE = re.compile(r"https?://\S+")
_PII_PATTERNS = re.compile(
    r"\b(password|credit card|card number|ssn|social security|"
    r"bank account|routing number|cvv|pin number)\b",
    re.IGNORECASE,
)
_POLICY_FABRICATION_PATTERNS = re.compile(
    r"\b(we will refund|refund of \$|credit your account|"
    r"we have issued|we can offer you|free month|compensation)\b",
    re.IGNORECASE,
)
_ACCOUNT_ACTION_PATTERNS = re.compile(
    r"\b(reset your password|changed your email|"
    r"unlocked your account|restored your|deleted your)\b",
    re.IGNORECASE,
)

GENERATION_PROMPT = """{voice_guidelines}

## Your Task
You are drafting a public Twitter reply as a SpotifyCares agent. Use the retrieved
historical conversations below as grounding -- base your suggestions on what has
worked before, not on invented information.

## Retrieved Similar Conversations (from past support interactions)
{retrieved_context}

## Current Customer Message
{customer_message}

## Classification
Route: {route} | Domain: {domain} | Risk: {risk_flag}

## Instructions
1. Address the customer's specific issue based on the classification.
2. Ground your response in the patterns from retrieved conversations.
3. Do NOT invent URLs, policies, refund amounts, or account actions.
4. Do NOT ask for passwords, credit card numbers, or sensitive PII in public.
5. If the issue requires account-specific action, suggest DM.
6. Stay within 280 characters.
7. Sign with /AI at the end.

Draft your reply (280 characters max):"""

DM_FALLBACK_TEMPLATE = (
    "Hi there! We'd like to help with that. "
    "Could you send us a DM with more details so we can look into it? /AI"
)


def _format_retrieved_context(retrieved: list[dict]) -> str:
    """Format retrieved conversations for the prompt."""
    if not retrieved:
        return "No similar historical conversations found."

    parts = []
    for i, r in enumerate(retrieved, 1):
        parts.append(
            f"[{i}] (similarity: {r['similarity_score']:.2f})\n"
            f"  Customer: {r['customer_msg'][:150]}\n"
            f"  Agent: {r['agent_response'][:200]}"
        )
    return "\n".join(parts)


def _safety_check(reply: str, retrieved: list[dict]) -> dict:
    """
    Run post-generation safety checks.

    Returns dict with:
      passed: bool
      violations: list of violation descriptions
    """
    violations = []

    # 1. Character limit
    if len(reply) > config.TWITTER_CHAR_LIMIT:
        violations.append(
            f"Exceeds {config.TWITTER_CHAR_LIMIT} char limit ({len(reply)} chars)"
        )

    # 2. Invented URLs -- any URL in reply must exist in retrieved context
    reply_urls = set(_URL_RE.findall(reply))
    if reply_urls:
        retrieved_urls = set()
        for r in retrieved:
            retrieved_urls.update(_URL_RE.findall(r.get("agent_response", "")))
            retrieved_urls.update(_URL_RE.findall(r.get("customer_msg", "")))
        invented = reply_urls - retrieved_urls
        if invented:
            violations.append(f"Invented URL(s): {invented}")

    # 3. PII requests in public
    if _PII_PATTERNS.search(reply):
        violations.append("Contains PII request in public")

    # 4. Policy fabrication
    if _POLICY_FABRICATION_PATTERNS.search(reply):
        violations.append("Contains fabricated policy/refund claim")

    # 5. Unauthorized account actions
    if _ACCOUNT_ACTION_PATTERNS.search(reply):
        violations.append("Claims unauthorized account action")

    return {"passed": len(violations) == 0, "violations": violations}


def generate_reply(
    customer_message: str,
    classification: dict,
    retrieved: list[dict],
    evidence_strength: str = "moderate",
    provider: str | None = None,
    model: str | None = None,
) -> dict:
    """
    Generate a support reply grounded in historical conversations.

    Args:
        customer_message: Cleaned customer tweet text.
        classification: Dict with route, domain, risk_flag from classifier.
        retrieved: List of retrieved similar conversations.
        evidence_strength: "strong", "moderate", or "weak".
        provider: LLM provider override.
        model: LLM model override.

    Returns:
        dict with: reply, safety_check, fallback_used, raw_reply
    """
    provider = provider or config.LLM_PROVIDER
    model = model or config.LLM_MODEL

    # If evidence is weak, use DM fallback directly
    if evidence_strength == "weak" and classification.get("risk_flag", "none") == "none":
        return {
            "reply": DM_FALLBACK_TEMPLATE,
            "safety_check": {"passed": True, "violations": []},
            "fallback_used": True,
            "raw_reply": DM_FALLBACK_TEMPLATE,
            "reason": "Weak retrieval evidence -- defaulting to DM redirect",
        }

    # Build prompt
    prompt = GENERATION_PROMPT.format(
        voice_guidelines=get_voice_prompt_section(),
        retrieved_context=_format_retrieved_context(retrieved),
        customer_message=customer_message,
        route=classification.get("route", "support"),
        domain=classification.get("domain", "unknown"),
        risk_flag=classification.get("risk_flag", "none"),
    )

    # Generate
    raw_reply = _generate_llm(prompt, provider, model)

    # Safety check
    safety = _safety_check(raw_reply, retrieved)

    # If safety fails, fall back
    if not safety["passed"]:
        return {
            "reply": DM_FALLBACK_TEMPLATE,
            "safety_check": safety,
            "fallback_used": True,
            "raw_reply": raw_reply,
            "reason": f"Safety violation(s): {'; '.join(safety['violations'])}",
        }

    # Truncate to char limit if slightly over (LLMs sometimes overshoot)
    if len(raw_reply) > config.TWITTER_CHAR_LIMIT:
        # Try to truncate at last sentence boundary
        truncated = raw_reply[: config.TWITTER_CHAR_LIMIT]
        last_period = truncated.rfind(".")
        last_excl = truncated.rfind("!")
        cut = max(last_period, last_excl)
        if cut > config.TWITTER_CHAR_LIMIT * 0.6:
            raw_reply = truncated[: cut + 1]
        else:
            raw_reply = truncated.rstrip() + "..."

    return {
        "reply": raw_reply,
        "safety_check": safety,
        "fallback_used": False,
        "raw_reply": raw_reply,
        "reason": "Generated from LLM with retrieved context",
    }


def _generate_llm(prompt: str, provider: str, model: str) -> str:
    """Call the LLM API to generate a reply."""
    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            return _template_fallback(prompt)

        api_key = config.LLM_API_KEY or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return _template_fallback(prompt)

        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a SpotifyCares Twitter support agent. Write concise replies under 280 characters."},
                    {"role": "user", "content": prompt},
                ],
                temperature=config.LLM_TEMPERATURE_GENERATE,
                max_tokens=150,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return _template_fallback(prompt)
    else:
        # Fallback: template-based response for testing
        return _template_fallback(prompt)


def _template_fallback(prompt: str) -> str:
    """Template-based reply for testing without API access."""
    # Extract some context from the prompt to make it slightly dynamic
    if "crash" in prompt.lower() or "freeze" in prompt.lower():
        return "Hi there! Sorry to hear that. Can you try logging out, restarting your device, and logging back in? Let us know how it goes /AI"
    elif "play" in prompt.lower() or "skip" in prompt.lower() or "song" in prompt.lower():
        return "We'd like to help! What device and Spotify version are you using? Keep us posted /AI"
    elif "account" in prompt.lower() or "login" in prompt.lower():
        return "We're here to help! Send us a DM with your account details and we'll look into it /AI"
    elif "charge" in prompt.lower() or "bill" in prompt.lower() or "refund" in prompt.lower():
        return "We understand your concern. Please send us a DM so we can look into this privately /AI"
    else:
        return DM_FALLBACK_TEMPLATE


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick test with template fallback
    test_class = {"route": "support", "domain": "app_device", "risk_flag": "none"}
    test_retrieved = [{"customer_msg": "my app crashes", "agent_response": "try restarting", "similarity_score": 0.8}]
    result = generate_reply(
        "My Spotify keeps crashing on iPhone",
        test_class,
        test_retrieved,
        evidence_strength="moderate",
    )
    print(json.dumps(result, indent=2))
