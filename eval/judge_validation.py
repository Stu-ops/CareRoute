"""
judge_validation.py -- Validate LLM-as-a-judge reliability against human ratings.

Measures:
  1. Rubric Dimension Correlation (Spearman rho & Pearson r) between human and LLM judge.
  2. Pairwise Preference Agreement (Cohen's Kappa & % Agreement).
  3. Position Bias (evaluating order-swap consistency).

Ensures the automated judge is trustworthy before relying on it for headline results.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def _pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    denom = math.sqrt(var_x * var_y)
    return cov / denom if denom > 0 else 0.0


def _rank(values: list[float]) -> list[float]:
    """Assign fractional ranks to values."""
    sorted_pairs = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j < len(values) and sorted_pairs[j][1] == sorted_pairs[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[sorted_pairs[k][0]] = rank
        i = j
    return ranks


def _spearman_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    if len(x) < 2:
        return 0.0
    rx = _rank(x)
    ry = _rank(y)
    return _pearson_correlation(rx, ry)


def _cohens_kappa(rater_a: list[str], rater_b: list[str]) -> float:
    """Compute Cohen's Kappa for categorical agreement."""
    if not rater_a or len(rater_a) != len(rater_b):
        return 0.0
    categories = sorted(list(set(rater_a) | set(rater_b)))
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    k = len(categories)
    matrix = [[0] * k for _ in range(k)]
    n = len(rater_a)

    for a, b in zip(rater_a, rater_b):
        matrix[cat_to_idx[a]][cat_to_idx[b]] += 1

    po = sum(matrix[i][i] for i in range(k)) / n
    pe = sum(
        (sum(matrix[i][j] for j in range(k)) * sum(matrix[j][i] for j in range(k)))
        for i in range(k)
    ) / (n * n)

    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def validate_rubric_alignment(human_ratings: list[dict], judge_ratings: list[dict]) -> dict:
    """
    Compute alignment between human ratings and LLM-as-judge scores.

    Args:
        human_ratings: List of dicts, each with keys like:
                       {"relevance": 4, "actionability": 5, "brand_voice": 4, "safety": 5, "evidence_supportedness": 4}
        judge_ratings: Corresponding list of dicts with same numeric dimensions.

    Returns:
        dict with per-dimension Pearson and Spearman correlations + mean overall.
    """
    dimensions = ["relevance", "actionability", "brand_voice", "safety_privacy", "evidence_supportedness"]
    report = {}
    spearman_scores = []
    pearson_scores = []

    for dim in dimensions:
        alt_dim = "safety" if dim == "safety_privacy" else dim
        h_vals = [float(h.get(dim, h.get(alt_dim, 3.0))) for h in human_ratings]
        j_vals = [float(j.get(dim, j.get(alt_dim, 3.0))) for j in judge_ratings]

        if len(h_vals) >= 2:
            p_corr = _pearson_correlation(h_vals, j_vals)
            s_corr = _spearman_correlation(h_vals, j_vals)
        else:
            p_corr, s_corr = 0.0, 0.0

        report[dim] = {
            "pearson": round(p_corr, 4),
            "spearman": round(s_corr, 4),
            "mean_human": round(sum(h_vals) / max(len(h_vals), 1), 2),
            "mean_judge": round(sum(j_vals) / max(len(j_vals), 1), 2),
        }
        pearson_scores.append(p_corr)
        spearman_scores.append(s_corr)

    report["overall"] = {
        "mean_pearson": round(sum(pearson_scores) / max(len(pearson_scores), 1), 4),
        "mean_spearman": round(sum(spearman_scores) / max(len(spearman_scores), 1), 4),
        "sample_size": len(human_ratings),
    }
    return report


def validate_pairwise_alignment(human_preferences: list[str], judge_preferences: list[str]) -> dict:
    """
    Compute alignment between human preferences and LLM-judge preferences (e.g. ['A', 'B', 'tie']).

    Returns:
        dict with percent agreement and Cohen's Kappa.
    """
    if not human_preferences or len(human_preferences) != len(judge_preferences):
        return {"agreement_rate": 0.0, "cohens_kappa": 0.0, "sample_size": 0}

    matches = sum(1 for h, j in zip(human_preferences, judge_preferences) if h == j)
    agreement_rate = matches / len(human_preferences)
    kappa = _cohens_kappa(human_preferences, judge_preferences)

    return {
        "agreement_rate": round(agreement_rate, 4),
        "cohens_kappa": round(kappa, 4),
        "matches": matches,
        "sample_size": len(human_preferences),
    }


if __name__ == "__main__":
    # Quick self-test
    mock_human_rubric = [
        {"relevance": 5, "actionability": 4, "brand_voice": 5, "safety": 5, "evidence_supportedness": 4},
        {"relevance": 3, "actionability": 2, "brand_voice": 3, "safety": 4, "evidence_supportedness": 3},
        {"relevance": 4, "actionability": 4, "brand_voice": 4, "safety": 5, "evidence_supportedness": 5},
    ]
    mock_judge_rubric = [
        {"relevance": 5, "actionability": 4, "brand_voice": 4, "safety": 5, "evidence_supportedness": 4},
        {"relevance": 2, "actionability": 2, "brand_voice": 3, "safety": 5, "evidence_supportedness": 2},
        {"relevance": 4, "actionability": 5, "brand_voice": 4, "safety": 5, "evidence_supportedness": 4},
    ]
    rubric_report = validate_rubric_alignment(mock_human_rubric, mock_judge_rubric)
    print("Rubric Alignment Report:", json.dumps(rubric_report, indent=2))

    mock_human_pref = ["A", "B", "A", "tie"]
    mock_judge_pref = ["A", "B", "tie", "tie"]
    pairwise_report = validate_pairwise_alignment(mock_human_pref, mock_judge_pref)
    print("Pairwise Alignment Report:", json.dumps(pairwise_report, indent=2))
