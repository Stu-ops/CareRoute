"""
run_eval.py -- Full evaluation harness.

Runs all systems on the frozen test set:
  1. Trivial baseline
  2. Simple baseline (TF-IDF + 1-NN)
  3. Full RAG agent

Computes all metrics, runs safety audit, and produces comparison tables.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402
from eval.baselines.trivial_baseline import TrivialBaseline  # noqa: E402
from eval.baselines.simple_baseline import SimpleBaseline  # noqa: E402
from eval.metrics import compute_all_metrics, classification_metrics, escalation_metrics  # noqa: E402
from eval.safety_audit import run_safety_audit  # noqa: E402


def load_eval_set(path: str | Path) -> list[dict]:
    """Load a labeled evaluation set (JSONL)."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def run_system(system, examples: list[dict], system_name: str) -> list[dict]:
    """Run a system on evaluation examples. Returns list of result dicts."""
    print(f"\n{'='*60}")
    print(f"Running: {system_name} ({len(examples)} examples)")
    print(f"{'='*60}")

    results = []
    start = time.time()
    for i, ex in enumerate(examples):
        text = ex.get("clean_text", ex.get("text", ""))
        thread_length = ex.get("thread_length", 1)

        result = system.process_message(text, thread_length=thread_length)
        results.append(result)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  {i+1}/{len(examples)} ({elapsed:.1f}s)")

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s ({elapsed/max(len(examples),1):.2f}s/example)")
    return results


def print_comparison_table(all_metrics: dict[str, dict]):
    """Print a formatted comparison table."""
    systems = list(all_metrics.keys())

    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")

    # Route accuracy
    print(f"\n{'Metric':<35}", end="")
    for sys in systems:
        print(f"{sys:<20}", end="")
    print()
    print("-" * (35 + 20 * len(systems)))

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
    ]

    for name, getter in metrics_to_show:
        print(f"{name:<35}", end="")
        for sys_name in systems:
            val = getter(all_metrics[sys_name])
            if isinstance(val, float):
                print(f"{val:<20.4f}", end="")
            else:
                print(f"{str(val):<20}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="Run full evaluation harness")
    parser.add_argument(
        "--test-set", default=str(config.GOLDEN_TEST_SET_JSONL),
        help="Path to the labeled test set (JSONL)"
    )
    parser.add_argument(
        "--output", default=str(config.RESULTS_DIR / "metrics"),
        help="Output directory for metric files"
    )
    parser.add_argument(
        "--systems", nargs="+", default=["trivial", "simple", "full"],
        help="Which systems to evaluate"
    )
    args = parser.parse_args()

    test_set_path = Path(args.test_set)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test set
    if not test_set_path.exists():
        print(f"ERROR: Test set not found at {test_set_path}")
        print("Create the golden test set first (see docs/annotation_guide.md)")
        sys.exit(1)

    examples = load_eval_set(test_set_path)
    print(f"Loaded {len(examples)} test examples from {test_set_path}")

    all_metrics = {}
    all_results = {}

    # 1. Trivial baseline
    if "trivial" in args.systems:
        trivial = TrivialBaseline(config.TRAINING_LABELS_JSONL)
        trivial_results = run_system(trivial, examples, "Trivial Baseline")
        trivial_metrics = compute_all_metrics(trivial_results, examples)
        trivial_metrics["safety"] = run_safety_audit(trivial_results)
        all_metrics["trivial"] = trivial_metrics
        all_results["trivial"] = trivial_results

    # 2. Simple baseline
    if "simple" in args.systems:
        simple = SimpleBaseline(config.TRAINING_LABELS_JSONL)
        simple.load_retrieval_index()
        simple_results = run_system(simple, examples, "Simple Baseline")
        simple_metrics = compute_all_metrics(simple_results, examples)
        simple_metrics["safety"] = run_safety_audit(simple_results)
        all_metrics["simple"] = simple_metrics
        all_results["simple"] = simple_results

    # 3. Full agent
    if "full" in args.systems:
        try:
            from src.agent import SpotifyCaresAgent
            agent = SpotifyCaresAgent(use_cache=True)
            full_results = run_system(agent, examples, "Full RAG Agent")
            full_metrics = compute_all_metrics(full_results, examples)
            full_metrics["safety"] = run_safety_audit(full_results)
            all_metrics["full"] = full_metrics
            all_results["full"] = full_results
        except Exception as e:
            print(f"WARNING: Could not run full agent: {e}")

    # Print comparison
    if all_metrics:
        print_comparison_table(all_metrics)

    # Save results
    for sys_name, metrics in all_metrics.items():
        # Remove individual audit details for the summary file
        save_metrics = {k: v for k, v in metrics.items()}
        if "safety" in save_metrics:
            save_metrics["safety"] = {
                k: v for k, v in save_metrics["safety"].items()
                if k != "individual_audits"
            }

        out_path = output_dir / f"{sys_name}_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save_metrics, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nSaved {sys_name} metrics -> {out_path}")

    # Save full results for further analysis
    for sys_name, results in all_results.items():
        out_path = output_dir / f"{sys_name}_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nEvaluation complete. Results in {output_dir}")


if __name__ == "__main__":
    main()
