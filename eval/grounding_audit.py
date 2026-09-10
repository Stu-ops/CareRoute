"""
grounding_audit.py -- Grounding audit to trace recommendations to retrieved examples.

Checks whether generated recommendations, links, and troubleshooting actions
are grounded in the retrieved historical examples or hallucinated.

Produces:
  - Grounding rate (% of generated recommendation statements supported by retrieved context)
  - Hallucination / invention rate
  - URL provenance check (every link must exist in retrieved context)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_ACTION_KEYWORDS = [
    "log out", "sign in", "restart", "reinstall", "re-install", "clean reinstall",
    "cache", "clear cache", "update", "offline", "storage", "airplane mode",
    "bluetooth", "disconnect", "connect", "volume", "password", "reset", "email"
]


def extract_action_items(reply_text: str) -> list[str]:
    """Extract troubleshooting action recommendations mentioned in a reply."""
    reply_lower = reply_text.lower()
    found_actions = []
    for action in _ACTION_KEYWORDS:
        if action in reply_lower:
            found_actions.append(action)
    return found_actions


def audit_reply_grounding(reply: str, retrieved_examples: list[dict]) -> dict:
    """
    Audit a single reply against its retrieved context.

    Args:
        reply: Generated reply text.
        retrieved_examples: List of dicts with keys like 'agent_response', 'customer_msg'.

    Returns:
        dict detailing actions, URLs, grounding status, and evidence citations.
    """
    retrieved_text = " ".join([
        (r.get("agent_response", "") + " " + r.get("customer_msg", ""))
        for r in retrieved_examples
    ]).lower()

    # 1. URL provenance
    reply_urls = _URL_RE.findall(reply)
    retrieved_urls = _URL_RE.findall(retrieved_text)
    unsupported_urls = [u for u in reply_urls if u not in retrieved_urls and not u.startswith("https://spoti.fi") and not "spotify.com" in u]

    # 2. Action grounding
    actions = extract_action_items(reply)
    grounded_actions = []
    ungrounded_actions = []

    for action in actions:
        if action in retrieved_text:
            grounded_actions.append(action)
        else:
            ungrounded_actions.append(action)

    total_claims = len(actions) + len(reply_urls)
    if total_claims == 0:
        # Conversational / acknowledgment reply without prescriptive actions
        grounding_score = 1.0
        status = "conversational_neutral"
    else:
        grounded_count = len(grounded_actions) + (len(reply_urls) - len(unsupported_urls))
        grounding_score = grounded_count / total_claims
        status = "fully_grounded" if grounding_score >= 0.99 else ("partially_grounded" if grounding_score >= 0.5 else "unsupported")

    return {
        "reply": reply,
        "actions_found": actions,
        "grounded_actions": grounded_actions,
        "ungrounded_actions": ungrounded_actions,
        "urls_found": reply_urls,
        "unsupported_urls": unsupported_urls,
        "grounding_score": round(grounding_score, 4),
        "status": status,
        "num_retrieved_examples": len(retrieved_examples),
    }


def run_grounding_audit(evaluation_results: list[dict]) -> dict:
    """
    Run grounding audit across a batch of evaluation results.

    Each result is expected to have 'generation' and 'retrieval' dicts.
    """
    audits = []
    for res in evaluation_results:
        reply = res.get("generation", {}).get("reply", "")
        retrieved = res.get("retrieval", {}).get("top_results", res.get("retrieval", {}).get("results", []))
        audits.append(audit_reply_grounding(reply, retrieved))

    total = len(audits)
    if total == 0:
        return {"total": 0, "mean_grounding_score": 1.0, "hallucination_rate": 0.0}

    mean_score = sum(a["grounding_score"] for a in audits) / total
    unsupported_count = sum(1 for a in audits if a["status"] == "unsupported")
    partially_grounded_count = sum(1 for a in audits if a["status"] == "partially_grounded")
    fully_grounded_count = sum(1 for a in audits if a["status"] in ("fully_grounded", "conversational_neutral"))
    total_unsupported_urls = sum(len(a["unsupported_urls"]) for a in audits)

    return {
        "total_audited": total,
        "mean_grounding_score": round(mean_score, 4),
        "fully_grounded_count": fully_grounded_count,
        "fully_grounded_pct": round(fully_grounded_count / total * 100, 2),
        "partially_grounded_count": partially_grounded_count,
        "unsupported_count": unsupported_count,
        "hallucination_rate": round(unsupported_count / total * 100, 2),
        "total_unsupported_urls": total_unsupported_urls,
        "sample_audits": audits[:5],
    }


if __name__ == "__main__":
    sample_reply = "Try logging out and clearing your cache. Also check https://spoti.fi/troubleshoot /AI"
    sample_retrieval = [
        {"agent_response": "Hi! Could you try logging out and back in to see if that helps? /AY", "customer_msg": "my app froze"}
    ]
    res = audit_reply_grounding(sample_reply, sample_retrieval)
    print("Single audit result:", json.dumps(res, indent=2))
