"""
extract_brand.py -- Filter the full TWCS CSV to a single brand's conversations.

Reads the ~516 MB CSV in streaming mode (never loads the full file into memory)
and writes all tweets that belong to the target brand's conversations to a JSONL file.

A tweet "belongs" to a brand's conversations if:
  1. The tweet's author IS the brand (outbound), OR
  2. The tweet is referenced by a brand tweet (via in_response_to_tweet_id), OR
  3. The tweet references a brand tweet (via response_tweet_id).

Because (2) and (3) require two passes, we:
  Pass 1 -- Collect all tweet IDs that the brand authored or directly references.
  Pass 2 -- Collect all tweets whose IDs are in that set.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402


def _parse_id_list(raw: str) -> list[str]:
    """Split a comma-separated ID field into a list of stripped IDs."""
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def extract_brand(
    csv_path: str | Path = config.TWCS_CSV_PATH,
    brand: str = config.TARGET_BRAND,
    output_path: str | Path = config.SPOTIFY_RAW_JSONL,
) -> dict:
    """
    Extract all tweets involved in *brand* conversations.

    Returns a summary dict with counts.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: discover relevant tweet IDs ──────────────────────────────
    print(f"[extract_brand] Pass 1 -- scanning for {brand} tweets ...")
    brand_tweet_ids: set[str] = set()  # IDs authored by brand
    related_ids: set[str] = set()       # IDs referenced by brand tweets
    total_rows = 0

    with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total_rows += 1
            author = row.get("author_id", "").strip()
            tid = row.get("tweet_id", "").strip()
            if author == brand:
                brand_tweet_ids.add(tid)
                # Collect tweets the brand was replying to
                for ref in _parse_id_list(row.get("in_response_to_tweet_id", "")):
                    related_ids.add(ref)
                # Collect tweets that reply to the brand
                for ref in _parse_id_list(row.get("response_tweet_id", "")):
                    related_ids.add(ref)
            if total_rows % 500_000 == 0:
                print(f"  ... {total_rows:,} rows scanned")

    target_ids = brand_tweet_ids | related_ids
    print(
        f"  Pass 1 done: {total_rows:,} rows, "
        f"{len(brand_tweet_ids):,} brand tweets, "
        f"{len(target_ids):,} target IDs"
    )

    # ── Pass 2: extract matching tweets ──────────────────────────────────
    print("[extract_brand] Pass 2 -- extracting matching tweets ...")
    written = 0
    with (
        open(csv_path, "r", encoding="utf-8", errors="replace") as fh,
        open(output_path, "w", encoding="utf-8") as out,
    ):
        reader = csv.DictReader(fh)
        for row in reader:
            tid = row.get("tweet_id", "").strip()
            if tid in target_ids:
                record = {
                    "tweet_id": tid,
                    "author_id": row.get("author_id", "").strip(),
                    "inbound": row.get("inbound", "").strip() == "True",
                    "created_at": row.get("created_at", "").strip(),
                    "text": row.get("text", ""),
                    "response_tweet_ids": _parse_id_list(
                        row.get("response_tweet_id", "")
                    ),
                    "in_response_to_tweet_id": row.get(
                        "in_response_to_tweet_id", ""
                    ).strip(),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    summary = {
        "total_rows_scanned": total_rows,
        "brand": brand,
        "brand_tweet_count": len(brand_tweet_ids),
        "target_id_count": len(target_ids),
        "tweets_written": written,
        "output_path": str(output_path),
    }
    print(f"[extract_brand] Done -- wrote {written:,} tweets -> {output_path}")
    return summary


# ─── CLI entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract a single brand from TWCS.")
    parser.add_argument("--csv", default=config.TWCS_CSV_PATH, help="Path to twcs.csv")
    parser.add_argument("--brand", default=config.TARGET_BRAND, help="Brand author_id")
    parser.add_argument(
        "--output", default=str(config.SPOTIFY_RAW_JSONL), help="Output JSONL path"
    )
    args = parser.parse_args()
    result = extract_brand(args.csv, args.brand, args.output)
    print(json.dumps(result, indent=2))
