"""
run_eval.py -- Comprehensive evaluation harness.

Runs all systems on the frozen test set:
  1. Trivial baseline (most-frequent class, canned template)
  2. Simple baseline (TF-IDF + Logistic Regression + 1-NN + rule-based router)
  3. Full RAG agent (hierarchical classification + thread-aware RAG + safety-checked generation + calibrated router)

Computes:
  - Classification metrics (Accuracy, Macro-F1 per layer with bootstrap CIs)
  - Escalation metrics (Precision, Recall, F1, False Negative Rate with bootstrap CIs)
  - Safety audit (PII, URL, Fabricated refund scanning)
  - Grounding audit (tracing recommendations & URLs to retrieved context)
  - Coverage-Risk Calibration on development set (with matplotlib curve plot)
  - Retrieval ablations (K=3 vs K=5 vs K=10, domain filter ON vs OFF)
  - Failure analysis breakdown (per-slice error rates & confusion matrix)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from eval.baselines.trivial_baseline import TrivialBaseline
from eval.baselines.simple_baseline import SimpleBaseline
from eval.metrics import compute_all_metrics
from eval.safety_audit import run_safety_audit
from eval.grounding_audit import run_grounding_audit
from eval.coverage_risk import compute_coverage_risk_curve, plot_coverage_risk_curve


def load_eval_set(path: str | Path) -> list[dict]:
    """Load a labeled evaluation set (JSONL)."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def run_system(
    system,
    examples: list[dict],
    system_name: str,
    force_refresh: bool = False,
) -> list[dict]:
    """Run a system on evaluation examples with thread context and isolation."""
    print(f"\n{'='*60}")
    print(f"Running: {system_name} ({len(examples)} examples)")
    print(f"{'='*60}")

    results = []
    start = time.time()
    for i, ex in enumerate(examples):
        text = ex.get("clean_text", ex.get("text", ""))
        thread_id = ex.get("thread_id")
        thread_context = ex.get("thread_context", "")
        thread_length = ex.get("thread_length", 1)

        result = system.process_message(
            text,
            thread_id=thread_id,
            thread_context=thread_context,
            thread_length=thread_length,
            force_refresh=force_refresh,
        )
        results.append(result)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  {i+1}/{len(examples)} ({elapsed:.1f}s)")

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s ({elapsed/max(len(examples),1):.2f}s/example)")
    return results


def run_ablations(agent_cls, dev_examples: list[dict], output_dir: Path) -> dict:
    """Run retrieval ablations on the development set."""
    print("\n" + "="*60)
    print("Running Ablations on Development Set")
    print("="*60)

    ablation_results = {}
    k_values = [3, 5, 10]

    for k in k_values:
        print(f"  Evaluating retrieval K={k}...")
        agent = agent_cls(retrieval_k=k, use_cache=False)
        res = run_system(agent, dev_examples, f"Ablation K={k}")
        metrics = compute_all_metrics(res, dev_examples)
        grounding = run_grounding_audit(res)
        ablation_results[f"k_{k}"] = {
            "retrieval_k": k,
            "route_macro_f1": metrics.get("route_classification", {}).get("macro_f1", 0),
            "domain_macro_f1": metrics.get("domain_classification", {}).get("macro_f1", 0),
            "escalation_recall": metrics.get("escalation", {}).get("recall", 0),
            "escalation_fnr": metrics.get("escalation", {}).get("false_negative_rate", 0),
            "grounding_score": grounding.get("mean_grounding_score", 0),
        }

    out_file = output_dir / "ablations.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"Saved ablations report -> {out_file}")
    return ablation_results


def run_calibration(agent, dev_examples: list[dict], output_dir: Path) -> dict:
    """Compute and plot coverage-risk calibration curve on development set."""
    print("\n" + "="*60)
    print("Running Coverage-Risk Calibration on Development Set")
    print("="*60)

    results = run_system(agent, dev_examples, "Calibration (Dev Set)")
    confidences = []
    correct_escalations = []

    for res, ex in zip(results, dev_examples):
        pred = res.get("escalation", {}).get("decision", "auto_handle")
        gold = ex.get("escalation_decision", "auto_handle")
        conf = res.get("escalation", {}).get("confidence", 0.5)
        confidences.append(float(conf))
        correct_escalations.append(pred == gold)

    curve_data = compute_coverage_risk_curve(confidences, correct_escalations)

    # Plot
    plot_path = output_dir.parent / "coverage_risk_curve.png"
    plot_coverage_risk_curve(curve_data, output_path=plot_path)

    # Save JSON curve
    json_path = output_dir / "calibration_curve.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(curve_data, f, indent=2)
    print(f"Saved calibration curve data -> {json_path}")
    return curve_data


def compute_failure_analysis(full_results: list[dict], examples: list[dict]) -> dict:
    """Compute sliced failure analysis across routes, domains, and risk flags."""
    slices = {
        "domain_errors": {},
        "risk_misclassifications": [],
        "escalation_false_negatives": [],
    }

    for res, ex in zip(full_results, examples):
        c_pred = res.get("classification", {})
        e_pred = res.get("escalation", {})
        text = ex.get("clean_text", ex.get("text", ""))

        # Check domain error
        gold_d = ex.get("domain")
        pred_d = c_pred.get("domain")
        if gold_d:
            if gold_d not in slices["domain_errors"]:
                slices["domain_errors"][gold_d] = {"total": 0, "correct": 0}
            slices["domain_errors"][gold_d]["total"] += 1
            if gold_d == pred_d:
                slices["domain_errors"][gold_d]["correct"] += 1

        # Check escalation false negative (escalate in gold, but auto_handle in pred)
        gold_esc = ex.get("escalation_decision")
        pred_esc = e_pred.get("decision")
        if gold_esc == "escalate" and pred_esc == "auto_handle":
            slices["escalation_false_negatives"].append({
                "text": text[:120],
                "gold_risk": ex.get("risk_flag"),
                "pred_risk": c_pred.get("risk_flag"),
                "pred_reason": e_pred.get("reason"),
            })

    # Compute accuracy per domain
    domain_accuracy = {}
    for dom, counts in slices["domain_errors"].items():
        domain_accuracy[dom] = round(counts["correct"] / max(counts["total"], 1), 4)
    slices["domain_accuracy"] = domain_accuracy

    return slices


def print_comparison_table(all_metrics: dict[str, dict]):
    """Print a formatted comparison table across evaluated systems."""
    systems = list(all_metrics.keys())

    print(f"\n{'='*85}")
    print("COMPARISON TABLE")
    print(f"{'='*85}")

    print(f"{'Metric':<35}", end="")
    for sys in systems:
        print(f"{sys:<22}", end="")
    print()
    print("-" * (35 + 22 * len(systems)))

    metrics_to_show = [
        ("Route Accuracy", lambda m: m.get("route_classification", {}).get("accuracy", "N/A")),
        ("Route Macro-F1", lambda m: m.get("route_classification", {}).get("macro_f1", "N/A")),
        ("Domain Macro-F1", lambda m: m.get("domain_classification", {}).get("macro_f1", "N/A")),
        ("Risk Macro-F1", lambda m: m.get("risk_classification", {}).get("macro_f1", "N/A")),
        ("Escalation Precision", lambda m: m.get("escalation", {}).get("precision", "N/A")),
        ("Escalation Recall", lambda m: m.get("escalation", {}).get("recall", "N/A")),
        ("Escalation F1", lambda m: m.get("escalation", {}).get("f1", "N/A")),
        ("Escalation FNR", lambda m: m.get("escalation", {}).get("false_negative_rate", "N/A")),
        ("Safety Violation %", lambda m: m.get("safety", {}).get("violation_rate", "N/A")),
        ("Grounding Score (0-1)", lambda m: m.get("grounding", {}).get("mean_grounding_score", "N/A")),
    ]

    for name, getter in metrics_to_show:
        print(f"{name:<35}", end="")
        for sys_name in systems:
            val = getter(all_metrics[sys_name])
            if isinstance(val, float):
                print(f"{val:<22.4f}", end="")
            else:
                print(f"{str(val):<22}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive evaluation harness")
    parser.add_argument(
        "--test-set", default=str(config.GOLDEN_TEST_SET_JSONL),
        help="Path to the labeled test set (JSONL)"
    )
    parser.add_argument(
        "--dev-set", default=str(config.DEV_SET_JSONL),
        help="Path to the labeled dev set (JSONL)"
    )
    parser.add_argument(
        "--output", default=str(config.RESULTS_DIR / "metrics"),
        help="Output directory for metric files"
    )
    parser.add_argument(
        "--systems", nargs="+", default=["trivial", "simple", "full"],
        help="Which systems to evaluate"
    )
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Force recomputation and bypass cache"
    )
    parser.add_argument(
        "--ablations", action="store_true",
        help="Run retrieval K ablations on the dev set"
    )
    parser.add_argument(
        "--calibration", action="store_true",
        help="Run coverage-risk calibration on the dev set"
    )
    parser.add_argument(
        "--subsample", type=int, default=None,
        help="Subsample N examples from the evaluation set (expected & encouraged by assignment)"
    )
    args = parser.parse_args()

    test_set_path = Path(args.test_set)
    dev_set_path = Path(args.dev_set)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not test_set_path.exists():
        print(f"ERROR: Test set not found at {test_set_path}")
        sys.exit(1)

    examples = load_eval_set(test_set_path)
    if args.subsample and args.subsample > 0 and args.subsample < len(examples):
        examples = examples[:args.subsample]
        print(f"Subsampled to {len(examples)} examples (as requested by --subsample)")
    print(f"Loaded {len(examples)} test examples from {test_set_path}")

    all_metrics = {}
    all_results = {}

    # 1. Trivial baseline
    if "trivial" in args.systems:
        trivial = TrivialBaseline(config.TRAINING_LABELS_JSONL)
        trivial_results = run_system(trivial, examples, "Trivial Baseline", force_refresh=args.force_refresh)
        trivial_metrics = compute_all_metrics(trivial_results, examples)
        trivial_metrics["safety"] = run_safety_audit(trivial_results)
        all_metrics["trivial"] = trivial_metrics
        all_results["trivial"] = trivial_results

    # 2. Simple baseline
    if "simple" in args.systems:
        simple = SimpleBaseline(config.TRAINING_LABELS_JSONL)
        simple.load_retrieval_index()
        simple_results = run_system(simple, examples, "Simple Baseline", force_refresh=args.force_refresh)
        simple_metrics = compute_all_metrics(simple_results, examples)
        simple_metrics["safety"] = run_safety_audit(simple_results)
        all_metrics["simple"] = simple_metrics
        all_results["simple"] = simple_results

    # 3. Full agent
    if "full" in args.systems:
        try:
            from src.agent import SpotifyCaresAgent
            agent = SpotifyCaresAgent(use_cache=(not args.force_refresh))
            full_results = run_system(agent, examples, "Full RAG Agent", force_refresh=args.force_refresh)
            full_metrics = compute_all_metrics(full_results, examples)
            full_metrics["safety"] = run_safety_audit(full_results)
            full_metrics["grounding"] = run_grounding_audit(full_results)
            all_metrics["full"] = full_metrics
            all_results["full"] = full_results

            # Failure Analysis
            failure_analysis = compute_failure_analysis(full_results, examples)
            fa_path = output_dir / "failure_analysis.json"
            with open(fa_path, "w", encoding="utf-8") as f:
                json.dump(failure_analysis, f, indent=2)
            print(f"Saved failure analysis breakdown -> {fa_path}")

            # Run calibration if dev set is present
            if args.calibration and dev_set_path.exists():
                dev_examples = load_eval_set(dev_set_path)
                run_calibration(agent, dev_examples, output_dir)

            # Run ablations if requested
            if args.ablations and dev_set_path.exists():
                dev_examples = load_eval_set(dev_set_path)
                run_ablations(SpotifyCaresAgent, dev_examples, output_dir)

        except Exception as e:
            print(f"WARNING: Could not run full agent: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison
    if all_metrics:
        print_comparison_table(all_metrics)

    # Save metrics JSONs
    for sys_name, metrics in all_metrics.items():
        save_metrics = {k: v for k, v in metrics.items()}
        if "safety" in save_metrics:
            save_metrics["safety"] = {
                k: v for k, v in save_metrics["safety"].items()
                if k != "individual_audits"
            }
        out_path = output_dir / f"{sys_name}_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save_metrics, f, indent=2, ensure_ascii=False, default=str)
        print(f"Saved {sys_name} metrics -> {out_path}")

    # Save full results JSONs
    for sys_name, results in all_results.items():
        out_path = output_dir / f"{sys_name}_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nEvaluation complete. Results saved in {output_dir}")


if __name__ == "__main__":
    main()
