"""
thread_builder.py -- Reconstruct multi-turn conversation threads from flat tweets.

Algorithm
---------
1. Load all tweets from the brand-extracted JSONL into a dict keyed by tweet_id.
2. Build a parent->children adjacency list using response_tweet_id / in_response_to_tweet_id.
3. Identify root tweets (no parent, or parent not in the dataset).
4. BFS from each root to collect a full thread.
5. Sort messages within each thread by timestamp.
6. Record quality metrics: branch counts, dangling references, orphan tweets.

Branch handling
---------------
All branches are preserved. A thread with N branches stores all N response variants.
The `branch_count` field records how many tweets in the thread have >1 response.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402


def _parse_timestamp(raw: str) -> datetime | None:
    """Parse Twitter-style timestamps. Returns None on failure."""
    try:
        return datetime.strptime(raw, "%a %b %d %H:%M:%S %z %Y")
    except (ValueError, TypeError):
        return None


def load_tweets(jsonl_path: str | Path) -> dict[str, dict]:
    """Load the brand-extracted JSONL into a tweet_id -> record dict."""
    tweets: dict[str, dict] = {}
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            tid = rec["tweet_id"]
            rec["_ts"] = _parse_timestamp(rec.get("created_at", ""))
            tweets[tid] = rec
    return tweets


def build_threads(
    input_path: str | Path = config.SPOTIFY_RAW_JSONL,
    output_path: str | Path = config.SPOTIFY_THREADS_JSONL,
) -> dict:
    """
    Reconstruct threads and write to JSONL.

    Returns a summary dict with quality metrics.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tweets = load_tweets(input_path)
    print(f"[thread_builder] Loaded {len(tweets):,} tweets")

    # ── Build adjacency ──────────────────────────────────────────────────
    children: dict[str, list[str]] = defaultdict(list)
    dangling_refs = 0

    for tid, rec in tweets.items():
        parent_id = rec.get("in_response_to_tweet_id", "").strip()
        if parent_id:
            if parent_id in tweets:
                children[parent_id].append(tid)
            else:
                dangling_refs += 1

    # ── Identify roots ───────────────────────────────────────────────────
    roots: list[str] = []
    for tid, rec in tweets.items():
        parent_id = rec.get("in_response_to_tweet_id", "").strip()
        if not parent_id or parent_id not in tweets:
            roots.append(tid)

    print(f"[thread_builder] {len(roots):,} root tweets, {dangling_refs:,} dangling refs")

    # ── BFS to build threads ─────────────────────────────────────────────
    visited: set[str] = set()
    threads: list[dict] = []
    orphan_count = 0

    for root in roots:
        if root in visited:
            continue
        thread_messages: list[dict] = []
        branch_count = 0
        queue: deque[str] = deque([root])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            if current in tweets:
                rec = tweets[current]
                thread_messages.append({
                    "tweet_id": rec["tweet_id"],
                    "author_id": rec["author_id"],
                    "inbound": rec["inbound"],
                    "text": rec["text"],
                    "created_at": rec["created_at"],
                })

                kids = children.get(current, [])
                if len(kids) > 1:
                    branch_count += 1
                for kid in kids:
                    if kid not in visited:
                        queue.append(kid)

        if not thread_messages:
            orphan_count += 1
            continue

        # Sort by timestamp within thread
        thread_messages.sort(
            key=lambda m: _parse_timestamp(m["created_at"]) or datetime.min
        )

        thread_id = thread_messages[0]["tweet_id"]  # root tweet ID
        threads.append({
            "thread_id": thread_id,
            "length": len(thread_messages),
            "branch_count": branch_count,
            "messages": thread_messages,
        })

    # Check for tweets not reached by any BFS
    unvisited = set(tweets.keys()) - visited
    if unvisited:
        print(f"[thread_builder] WARNING: {len(unvisited)} tweets not reached by BFS")

    # ── Write output ─────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as out:
        for thread in threads:
            out.write(json.dumps(thread, ensure_ascii=False) + "\n")

    # ── Quality metrics ──────────────────────────────────────────────────
    lengths = [t["length"] for t in threads]
    branches = [t["branch_count"] for t in threads]
    multi_turn = sum(1 for l in lengths if l >= 3)
    rich = sum(1 for l in lengths if l >= 5)

    summary = {
        "total_threads": len(threads),
        "total_messages": sum(lengths),
        "dangling_references": dangling_refs,
        "orphan_tweets": orphan_count,
        "unvisited_tweets": len(unvisited),
        "branching_threads": sum(1 for b in branches if b > 0),
        "multi_turn_threads_gte3": multi_turn,
        "multi_turn_pct": round(100 * multi_turn / max(len(threads), 1), 1),
        "rich_threads_gte5": rich,
        "rich_pct": round(100 * rich / max(len(threads), 1), 1),
        "thread_length_distribution": {
            str(k): sum(1 for l in lengths if l == k)
            for k in sorted(set(lengths))
            if k <= 20 or sum(1 for l in lengths if l == k) > 5
        },
        "output_path": str(output_path),
    }
    print(f"[thread_builder] Done -- {len(threads):,} threads -> {output_path}")
    for key, val in summary.items():
        if key not in ("thread_length_distribution", "output_path"):
            print(f"  {key}: {val}")
    return summary


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = build_threads()
    print(json.dumps(result, indent=2))
