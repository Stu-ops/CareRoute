"""
llm_judge.py -- LLM-as-judge evaluation with 5-dimension rubric + pairwise comparison.

Headline reply evaluation dimensions:
  1. Relevance (1-5)
  2. Actionability (1-5)
  3. Brand Voice (1-5)
  4. Safety/Privacy (1-5)
  5. Evidence-Supportedness (1-5)

Also supports pairwise comparison between two systems.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402

RUBRIC_PROMPT = """You are evaluating a customer support reply from SpotifyCares on Twitter.

## Customer Message
{customer_message}

## Agent Reply
{agent_reply}

## Retrieved Context (historical conversations used as grounding)
{retrieved_context}

## Evaluation Rubric
Rate the reply on each dimension from 1 (worst) to 5 (best):

1. **Relevance** (1-5): Does the reply address the customer's actual issue?
   1=completely off-topic, 3=partially relevant, 5=directly addresses the issue

2. **Actionability** (1-5): Does the reply give the customer a clear next step?
   1=no next step, 3=vague suggestion, 5=concrete actionable step

3. **Brand Voice** (1-5): Does it match SpotifyCares' warm, helpful, concise tone?
   1=robotic/cold, 3=acceptable, 5=sounds exactly like SpotifyCares

4. **Safety/Privacy** (1-5): No invented policies, links, refunds, or PII requests?
   1=contains fabricated claims or PII requests, 3=minor concern, 5=completely safe

5. **Evidence-Supportedness** (1-5): Are recommendations supported by the retrieved context?
   1=entirely hallucinated, 3=partially supported, 5=fully grounded in retrieved examples

Respond in JSON format ONLY:
{{
  "relevance": <1-5>,
  "actionability": <1-5>,
  "brand_voice": <1-5>,
  "safety_privacy": <1-5>,
  "evidence_supportedness": <1-5>,
  "reasoning": "<brief justification>"
}}"""

PAIRWISE_PROMPT = """You are comparing two customer support replies for the same customer message.

## Customer Message
{customer_message}

## Reply A
{reply_a}

## Reply B
{reply_b}

Which reply is better overall for a SpotifyCares Twitter support agent? Consider:
- Does it address the customer's issue?
- Does it provide a clear next step?
- Does it match the brand's warm, helpful tone?
- Is it safe (no fabricated info)?

Respond in JSON format ONLY:
{{
  "winner": "A" or "B" or "tie",
  "reasoning": "<brief justification>"
}}"""


def judge_single(
    customer_message: str,
    agent_reply: str,
    retrieved_context: str = "",
    provider: str | None = None,
    model: str | None = None,
) -> dict:
    """
    Judge a single reply on the 5-dimension rubric.

    Returns dict with scores per dimension + reasoning.
    """
    provider = provider or config.LLM_PROVIDER
    model = model or config.LLM_JUDGE_MODEL

    prompt = RUBRIC_PROMPT.format(
        customer_message=customer_message,
        agent_reply=agent_reply,
        retrieved_context=retrieved_context or "Not available",
    )

    raw = _call_llm(prompt, provider, model)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "relevance": 3, "actionability": 3, "brand_voice": 3,
            "safety_privacy": 3, "evidence_supportedness": 3,
            "reasoning": f"Parse error: {raw[:100]}",
        }

    # Validate and clamp scores
    for dim in ["relevance", "actionability", "brand_voice", "safety_privacy", "evidence_supportedness"]:
        result[dim] = max(1, min(5, int(result.get(dim, 3))))

    result["average"] = round(
        sum(result[d] for d in ["relevance", "actionability", "brand_voice",
                                 "safety_privacy", "evidence_supportedness"]) / 5, 2
    )
    return result


def judge_pairwise(
    customer_message: str,
    reply_a: str,
    reply_b: str,
    randomize_position: bool = True,
    provider: str | None = None,
    model: str | None = None,
) -> dict:
    """
    Pairwise comparison: which reply is better?

    Position is randomized to avoid position bias.
    Returns dict with winner (normalized back to original order) + reasoning.
    """
    provider = provider or config.LLM_PROVIDER
    model = model or config.LLM_JUDGE_MODEL

    # Randomize position to mitigate position bias
    swapped = False
    if randomize_position and random.random() > 0.5:
        reply_a, reply_b = reply_b, reply_a
        swapped = True

    prompt = PAIRWISE_PROMPT.format(
        customer_message=customer_message,
        reply_a=reply_a,
        reply_b=reply_b,
    )

    raw = _call_llm(prompt, provider, model)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"winner": "tie", "reasoning": f"Parse error: {raw[:100]}"}

    # Un-swap if needed
    winner = result.get("winner", "tie")
    if swapped:
        if winner == "A":
            winner = "B"
        elif winner == "B":
            winner = "A"

    return {
        "winner": winner,
        "reasoning": result.get("reasoning", ""),
        "position_swapped": swapped,
    }


def _call_llm(prompt: str, provider: str, model: str) -> str:
    """Call LLM API."""
    if provider == "openai":
        try:
            from openai import OpenAI
            api_key = config.LLM_API_KEY or os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                return json.dumps({"relevance": 3, "actionability": 3, "brand_voice": 3,
                                    "safety_privacy": 3, "evidence_supportedness": 3,
                                    "reasoning": "No API key"})
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Respond only in valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return json.dumps({"error": str(e)})
    else:
        # Fallback: return neutral scores
        return json.dumps({
            "relevance": 3, "actionability": 3, "brand_voice": 3,
            "safety_privacy": 4, "evidence_supportedness": 3,
            "reasoning": "Fallback judge (no LLM API configured)",
        })


def run_judge_evaluation(
    eval_results: list[dict],
    gold_labels: list[dict],
    max_examples: int | None = None,
) -> dict:
    """
    Run the LLM judge on evaluation results.

    Returns aggregate scores per dimension.
    """
    examples = list(zip(eval_results, gold_labels))
    if max_examples:
        examples = examples[:max_examples]

    all_scores = []
    for result, gold in examples:
        customer_msg = result["input"]["clean_text"]
        agent_reply = result["generation"]["reply"]
        retrieved = "\n".join(
            f"- {r['agent_response'][:100]}"
            for r in result.get("retrieval", {}).get("top_results", [])
        )

        scores = judge_single(customer_msg, agent_reply, retrieved)
        all_scores.append(scores)

    # Aggregate
    dims = ["relevance", "actionability", "brand_voice", "safety_privacy", "evidence_supportedness"]
    aggregated = {}
    for dim in dims:
        values = [s[dim] for s in all_scores]
        import numpy as np
        aggregated[dim] = {
            "mean": round(float(np.mean(values)), 2),
            "std": round(float(np.std(values)), 2),
            "min": min(values),
            "max": max(values),
        }

    avg_scores = [s.get("average", 3.0) for s in all_scores]
    aggregated["overall_average"] = {
        "mean": round(float(np.mean(avg_scores)), 2),
        "std": round(float(np.std(avg_scores)), 2),
    }
    aggregated["n_examples"] = len(all_scores)
    aggregated["individual_scores"] = all_scores

    return aggregated


if __name__ == "__main__":
    # Quick test
    score = judge_single(
        "My Spotify keeps crashing",
        "Hi there! Sorry to hear that. Try logging out, restarting your device, and logging back in. Let us know how it goes /AI",
    )
    print(json.dumps(score, indent=2))
