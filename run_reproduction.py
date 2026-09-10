"""
run_reproduction.py -- Single-command reproduction script.

Usage:
  python run_reproduction.py --cached    # Use pre-computed outputs (no API key needed)
  python run_reproduction.py --live      # Run with live API calls (needs API key)

Reproduces headline results in < 15 minutes using supplied processed data.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def check_dependencies():
    """Check that required packages are installed."""
    missing = []
    for pkg in ["numpy", "sklearn"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Missing required packages: {missing}")
        print("Run: pip install -r requirements.txt")
        return False
    return True


def check_processed_data():
    """Check that processed data files exist."""
    required = [
        config.SPOTIFY_THREADS_JSONL,
        config.SPLIT_MANIFEST_JSON,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("Missing processed data files:")
        for p in missing:
            print(f"  {p}")
        print("\nRun the data pipeline first:")
        print("  python src/data_pipeline/extract_brand.py")
        print("  python src/data_pipeline/thread_builder.py")
        print("  python src/data_pipeline/text_cleaner.py")
        print("  python src/data_pipeline/split_manifest.py")
        return False
    return True


def run_data_pipeline():
    """Run the full data pipeline if processed data doesn't exist."""
    print("="*60)
    print("Step 1: Data Pipeline")
    print("="*60)

    if config.SPOTIFY_RAW_JSONL.exists():
        print(f"  Brand extraction: CACHED ({config.SPOTIFY_RAW_JSONL})")
    else:
        print("  Running brand extraction...")
        from src.data_pipeline.extract_brand import extract_brand
        extract_brand()

    if config.SPOTIFY_THREADS_JSONL.exists():
        print(f"  Thread building: CACHED ({config.SPOTIFY_THREADS_JSONL})")
    else:
        print("  Running thread builder...")
        from src.data_pipeline.thread_builder import build_threads
        build_threads()

        print("  Running text cleaner...")
        from src.data_pipeline.text_cleaner import clean_threads
        clean_threads()

    if config.SPLIT_MANIFEST_JSON.exists():
        print(f"  Split manifest: CACHED ({config.SPLIT_MANIFEST_JSON})")
    else:
        print("  Running split manifest...")
        from src.data_pipeline.split_manifest import create_split_manifest
        create_split_manifest()


def run_evaluation(cached: bool = True, subsample: int | None = None):
    """Run evaluation with baselines."""
    print("\n" + "="*60)
    print("Step 2: Evaluation")
    print("="*60)

    test_set_path = config.GOLDEN_TEST_SET_JSONL
    if not test_set_path.exists():
        print(f"  WARNING: Golden test set not found at {test_set_path}")
        print("  Creating a sample evaluation set from test-split threads...")
        _create_sample_eval_set(test_set_path)

    print(f"  Running evaluation harness...")
    from eval.run_eval import main as run_eval_main
    sys.argv = [
        "run_eval.py",
        "--test-set", str(test_set_path),
        "--output", str(config.RESULTS_DIR / "metrics"),
        "--systems", "trivial", "simple", "full",
    ]
    if not cached:
        sys.argv.append("--force-refresh")
    if subsample:
        sys.argv.extend(["--subsample", str(subsample)])
    run_eval_main()


def _create_sample_eval_set(output_path: Path):
    """
    Create a sample evaluation set from test-split threads for demonstration.
    NOTE: This is NOT the hand-labeled golden set. It provides structure only.
    Real labels must be created manually following docs/annotation_guide.md.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = config.SPLIT_MANIFEST_JSON
    threads_path = config.SPOTIFY_THREADS_JSONL

    if not manifest_path.exists() or not threads_path.exists():
        print("  Cannot create sample eval set -- processed data missing")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    examples = []
    with open(threads_path, "r", encoding="utf-8") as f:
        for line in f:
            thread = json.loads(line)
            thread_id = thread["thread_id"]
            if manifest.get(thread_id) != "test":
                continue

            # Take inbound messages as eval examples
            for msg in thread["messages"]:
                if msg["inbound"] and len(msg.get("clean_text", msg["text"])) > 20:
                    examples.append({
                        "tweet_id": msg["tweet_id"],
                        "thread_id": thread_id,
                        "text": msg["text"],
                        "clean_text": msg.get("clean_text", msg["text"]),
                        "thread_length": thread["length"],
                        # Placeholder labels -- must be replaced with human labels
                        "route": "support",
                        "domain": None,
                        "risk_flag": "none",
                        "escalation_decision": "auto_handle",
                        "escalation_reason": "placeholder -- needs manual labeling",
                        "difficulty": "medium",
                        "notes": "AUTO-GENERATED PLACEHOLDER -- replace with human labels",
                    })

            if len(examples) >= 200:
                break

    # Write
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples[:200]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"  Created sample eval set: {len(examples[:200])} examples -> {output_path}")
    print("  WARNING: These are PLACEHOLDER labels. Replace with hand-labeled data.")


def main():
    parser = argparse.ArgumentParser(description="Reproduce headline results")
    parser.add_argument("--cached", action="store_true", help="Use cached outputs (no API key needed)")
    parser.add_argument("--live", action="store_true", help="Run with live API calls")
    parser.add_argument(
        "--subsample", type=int, default=None,
        help="Subsample N examples from the test set (per assignment rule: subsample encouraged)"
    )
    args = parser.parse_args()

    print("SpotifyCares AI Support Agent -- Reproduction Script")
    print("="*60)

    start = time.time()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Run data pipeline
    run_data_pipeline()

    # Run evaluation
    run_evaluation(cached=(not args.live), subsample=args.subsample)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Reproduction complete in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Results in: {config.RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
