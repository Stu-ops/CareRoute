"""
coverage_risk.py -- Coverage-risk curve for escalation threshold calibration.

At each confidence threshold T:
  coverage = fraction of messages the agent auto-handles
  risk = error rate on auto-handled messages

This replaces trusting a single raw confidence threshold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402


def compute_coverage_risk_curve(
    confidences: list[float],
    correct: list[bool],
    thresholds: list[float] | None = None,
) -> dict:
    """
    Compute coverage-risk curve.

    Args:
        confidences: Model confidence for each example.
        correct: Whether the model's prediction was correct for each example.
        thresholds: Optional list of thresholds to evaluate.

    Returns:
        dict with curve data and recommended threshold.
    """
    if thresholds is None:
        thresholds = [round(t * 0.05, 2) for t in range(1, 20)]  # 0.05 to 0.95

    curve = []
    for t in thresholds:
        # Auto-handle = confidence >= threshold
        auto_mask = [c >= t for c in confidences]
        n_auto = sum(auto_mask)
        coverage = n_auto / max(len(confidences), 1)

        if n_auto > 0:
            errors = sum(1 for m, c in zip(auto_mask, correct) if m and not c)
            risk = errors / n_auto
        else:
            risk = 0.0

        curve.append({
            "threshold": t,
            "coverage": round(coverage, 4),
            "risk": round(risk, 4),
            "n_auto_handled": n_auto,
            "n_errors": sum(1 for m, c in zip(auto_mask, correct) if m and not c),
        })

    # Find recommended threshold: highest coverage with risk < 10%
    recommended = None
    for point in curve:
        if point["risk"] <= 0.10 and point["coverage"] > 0:
            if recommended is None or point["coverage"] > recommended["coverage"]:
                recommended = point

    return {
        "curve": curve,
        "recommended_threshold": recommended,
        "n_examples": len(confidences),
        "target_max_risk": 0.10,
    }


def plot_coverage_risk(curve_data: dict, output_path: str | Path | None = None):
    """Generate and optionally save a coverage-risk curve plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[coverage_risk] matplotlib not installed, skipping plot")
        return

    curve = curve_data["curve"]
    coverages = [p["coverage"] for p in curve]
    risks = [p["risk"] for p in curve]
    thresholds = [p["threshold"] for p in curve]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(thresholds, coverages, "b-o", label="Coverage", linewidth=2)
    ax1.set_xlabel("Confidence Threshold", fontsize=12)
    ax1.set_ylabel("Coverage (fraction auto-handled)", color="b", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(thresholds, risks, "r-s", label="Risk (error rate)", linewidth=2)
    ax2.set_ylabel("Risk (error rate on auto-handled)", color="r", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.axhline(y=0.10, color="r", linestyle="--", alpha=0.5, label="10% risk target")

    # Mark recommended threshold
    rec = curve_data.get("recommended_threshold")
    if rec:
        ax1.axvline(x=rec["threshold"], color="g", linestyle="--", alpha=0.7,
                     label=f"Recommended: T={rec['threshold']}")

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3)
    plt.title("Coverage-Risk Curve: Escalation Threshold Calibration", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        print(f"[coverage_risk] Plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    n = 200
    confs = np.random.beta(3, 1.5, n).tolist()
    correct = [c > 0.4 + np.random.normal(0, 0.15) for c in confs]

    result = compute_coverage_risk_curve(confs, correct)
    print(json.dumps({k: v for k, v in result.items() if k != "curve"}, indent=2))
    print(f"\nCurve has {len(result['curve'])} points")

    plot_coverage_risk(result, config.RESULTS_DIR / "plots" / "coverage_risk_demo.png")
