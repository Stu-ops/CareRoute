"""
split_manifest.py -- Create a thread-level chronological train/dev/test split.

Rules (from the approved plan):
  1. Split by entire conversation thread -- never within a thread.
  2. Use chronological ordering: older threads for training, newer for dev/test.
  3. Boundaries:
       Training : threads whose earliest message is before TRAIN_CUTOFF
       Dev      : threads starting between TRAIN_CUTOFF and DEV_CUTOFF
       Test     : threads starting at or after DEV_CUTOFF
  4. Automated leak check: verify zero tweet/thread overlap across splits.

Output: data/splits/manifest.json
  {
    "thread_id_1": "train",
    "thread_id_2": "dev",
    ...
  }
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402


def _parse_ts(raw: str) -> datetime | None:
    try:
        return datetime.strptime(raw, "%a %b %d %H:%M:%S %z %Y")
    except (ValueError, TypeError):
        return None


def _earliest_timestamp(thread: dict) -> datetime | None:
    """Return the earliest timestamp in a thread's messages."""
    timestamps = []
    for msg in thread.get("messages", []):
        ts = _parse_ts(msg.get("created_at", ""))
        if ts:
            timestamps.append(ts)
    return min(timestamps) if timestamps else None


def create_split_manifest(
    threads_path: str | Path = config.SPOTIFY_THREADS_JSONL,
    output_path: str | Path = config.SPLIT_MANIFEST_JSON,
    train_cutoff: str = config.TRAIN_CUTOFF,
    dev_cutoff: str = config.DEV_CUTOFF,
) -> dict:
    """
    Assign each thread to train / dev / test based on its earliest timestamp.

    Returns summary dict with split sizes and leak-check results.
    """
    threads_path = Path(threads_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse cutoff dates (timezone-aware, UTC)
    from datetime import timezone

    train_cut = datetime.strptime(train_cutoff, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    dev_cut = datetime.strptime(dev_cutoff, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )

    manifest: dict[str, str] = {}
    split_counts = Counter()
    no_timestamp_count = 0

    # Also collect tweet_ids per split for leak check
    tweet_ids_by_split: dict[str, set[str]] = {
        "train": set(),
        "dev": set(),
        "test": set(),
    }

    with open(threads_path, "r", encoding="utf-8") as fh:
        for line in fh:
            thread = json.loads(line)
            thread_id = thread["thread_id"]
            earliest = _earliest_timestamp(thread)

            if earliest is None:
                # No parseable timestamp -- default to train
                split = "train"
                no_timestamp_count += 1
            elif earliest < train_cut:
                split = "train"
            elif earliest < dev_cut:
                split = "dev"
            else:
                split = "test"

            manifest[thread_id] = split
            split_counts[split] += 1

            for msg in thread.get("messages", []):
                tweet_ids_by_split[split].add(msg["tweet_id"])

    # ── Leak check ───────────────────────────────────────────────────────
    train_dev_overlap = tweet_ids_by_split["train"] & tweet_ids_by_split["dev"]
    train_test_overlap = tweet_ids_by_split["train"] & tweet_ids_by_split["test"]
    dev_test_overlap = tweet_ids_by_split["dev"] & tweet_ids_by_split["test"]

    leak_detected = bool(train_dev_overlap or train_test_overlap or dev_test_overlap)

    if leak_detected:
        print("[split_manifest] [FAIL] LEAK DETECTED!")
        print(f"  train∩dev: {len(train_dev_overlap)} tweets")
        print(f"  train∩test: {len(train_test_overlap)} tweets")
        print(f"  dev∩test: {len(dev_test_overlap)} tweets")
    else:
        print("[split_manifest] [OK] No leakage -- zero tweet overlap across splits")

    # ── Write manifest ───────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(manifest, out, indent=2, ensure_ascii=False)

    summary = {
        "total_threads": len(manifest),
        "train_threads": split_counts["train"],
        "dev_threads": split_counts["dev"],
        "test_threads": split_counts["test"],
        "train_tweets": len(tweet_ids_by_split["train"]),
        "dev_tweets": len(tweet_ids_by_split["dev"]),
        "test_tweets": len(tweet_ids_by_split["test"]),
        "no_timestamp_threads": no_timestamp_count,
        "leak_detected": leak_detected,
        "train_dev_overlap": len(train_dev_overlap),
        "train_test_overlap": len(train_test_overlap),
        "dev_test_overlap": len(dev_test_overlap),
        "cutoffs": {"train": train_cutoff, "dev": dev_cutoff},
        "output_path": str(output_path),
    }

    print(f"[split_manifest] Split sizes:")
    print(f"  train: {split_counts['train']:,} threads ({len(tweet_ids_by_split['train']):,} tweets)")
    print(f"  dev:   {split_counts['dev']:,} threads ({len(tweet_ids_by_split['dev']):,} tweets)")
    print(f"  test:  {split_counts['test']:,} threads ({len(tweet_ids_by_split['test']):,} tweets)")
    return summary


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = create_split_manifest()
    print(json.dumps(result, indent=2))
