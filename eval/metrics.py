"""
metrics.py -- Automated evaluation metrics with bootstrap confidence intervals.

Classification: Accuracy, Macro-F1, Per-class F1, confusion matrix, Cohen's Kappa
Reply: BLEU-4, ROUGE-L, BERTScore (secondary/diagnostic only)
Escalation: Precision, Recall, F1, False Negative Rate
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402


def _bootstrap_ci(
    values: list[float], n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval. Returns (mean, lower, upper)."""
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))
    means = sorted(means)
    alpha = (1 - ci) / 2
    lower = means[int(alpha * n_bootstrap)]
    upper = means[int((1 - alpha) * n_bootstrap)]
    return float(np.mean(arr)), float(lower), float(upper)


# ── Classification Metrics ───────────────────────────────────────────────────

def classification_metrics(
    y_true: list[str], y_pred: list[str], label_name: str = "classification"
) -> dict:
    """
    Compute classification metrics: accuracy, macro-F1, per-class F1, confusion matrix, Kappa.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report,
        confusion_matrix, cohen_kappa_score,
    )

    labels = sorted(set(y_true) | set(y_pred))

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Bootstrap CI for accuracy
    correct = [1 if t == p else 0 for t, p in zip(y_true, y_pred)]
    acc_mean, acc_lo, acc_hi = _bootstrap_ci(correct, config.BOOTSTRAP_N)

    return {
        "label_name": label_name,
        "n_examples": len(y_true),
        "accuracy": round(accuracy, 4),
        "accuracy_ci": [round(acc_lo, 4), round(acc_hi, 4)],
        "macro_f1": round(macro_f1, 4),
        "cohen_kappa": round(kappa, 4),
        "per_class": {
            k: {
                "precision": round(v.get("precision", 0), 4),
                "recall": round(v.get("recall", 0), 4),
                "f1-score": round(v.get("f1-score", 0), 4),
                "support": v.get("support", 0),
            }
            for k, v in per_class.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
    }


# ── Escalation Metrics ───────────────────────────────────────────────────────

def escalation_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """
    Compute escalation-specific metrics.
    Maps decisions to binary: escalate/clarify -> positive, auto_handle -> negative.
    """
    # Binary: escalate (or clarify) vs auto_handle
    y_true_bin = [1 if y in ("escalate", "clarify") else 0 for y in y_true]
    y_pred_bin = [1 if y in ("escalate", "clarify") else 0 for y in y_pred]

    tp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 0 and p == 0)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    fnr = fn / max(fn + tp, 1)  # False Negative Rate (missed escalations)

    return {
        "n_examples": len(y_true),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_negative_rate": round(fnr, 4),
        "note": "FNR is the critical metric -- missed escalations are dangerous",
    }


# ── Reply Metrics (Secondary/Diagnostic) ────────────────────────────────────

def reply_metrics_bleu_rouge(references: list[str], hypotheses: list[str]) -> dict:
    """
    Compute BLEU-4 and ROUGE-L. These are SECONDARY metrics only.
    Many valid replies exist for one customer message.
    """
    results = {"note": "SECONDARY METRICS -- many valid replies exist for one tweet"}

    # BLEU
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoother = SmoothingFunction().method1
        bleu_scores = []
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = ref.lower().split()
            hyp_tokens = hyp.lower().split()
            if hyp_tokens:
                score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother)
                bleu_scores.append(score)
        if bleu_scores:
            mean, lo, hi = _bootstrap_ci(bleu_scores, config.BOOTSTRAP_N)
            results["bleu4"] = {"mean": round(mean, 4), "ci": [round(lo, 4), round(hi, 4)]}
    except ImportError:
        results["bleu4"] = {"error": "nltk not installed"}

    # ROUGE-L
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores = []
        for ref, hyp in zip(references, hypotheses):
            score = scorer.score(ref, hyp)
            rouge_scores.append(score["rougeL"].fmeasure)
        if rouge_scores:
            mean, lo, hi = _bootstrap_ci(rouge_scores, config.BOOTSTRAP_N)
            results["rougeL"] = {"mean": round(mean, 4), "ci": [round(lo, 4), round(hi, 4)]}
    except ImportError:
        results["rougeL"] = {"error": "rouge-score not installed"}

    return results


def compute_all_metrics(
    eval_results: list[dict],
    gold_labels: list[dict],
) -> dict:
    """
    Compute all metrics from evaluation results vs gold labels.

    Args:
        eval_results: List of agent output dicts (from process_message).
        gold_labels: List of gold label dicts.

    Returns:
        dict with all metric groups.
    """
    # Classification metrics per layer
    route_true = [g.get("route", "support") for g in gold_labels]
    route_pred = [r["classification"]["route"] for r in eval_results]

    domain_pairs = [
        (g.get("domain"), r["classification"].get("domain"))
        for g, r in zip(gold_labels, eval_results)
        if g.get("route") == "support" and g.get("domain")
    ]
    domain_true = [p[0] for p in domain_pairs]
    domain_pred = [p[1] or "unknown" for p in domain_pairs]

    risk_true = [g.get("risk_flag", "none") for g in gold_labels]
    risk_pred = [r["classification"].get("risk_flag", "none") for r in eval_results]

    # Escalation
    esc_true = [g.get("escalation_decision", "auto_handle") for g in gold_labels]
    esc_pred = [r["escalation"]["decision"] for r in eval_results]

    # Reply metrics (using historical agent response as reference)
    references = [g.get("reference_reply", "") for g in gold_labels if g.get("reference_reply")]
    hypotheses = [r["generation"]["reply"] for r, g in zip(eval_results, gold_labels) if g.get("reference_reply")]

    metrics = {
        "route_classification": classification_metrics(route_true, route_pred, "route"),
        "domain_classification": classification_metrics(domain_true, domain_pred, "domain") if domain_true else {},
        "risk_classification": classification_metrics(risk_true, risk_pred, "risk_flag"),
        "escalation": escalation_metrics(esc_true, esc_pred),
    }

    if references and hypotheses:
        metrics["reply_diagnostic"] = reply_metrics_bleu_rouge(references, hypotheses)

    return metrics


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick demo
    y_true = ["support", "support", "feedback", "support", "abuse_spam"]
    y_pred = ["support", "feedback", "feedback", "support", "support"]
    result = classification_metrics(y_true, y_pred, "route")
    print(json.dumps(result, indent=2))
