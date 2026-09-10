"""
text_cleaner.py -- Normalize tweet text for downstream processing.

Operations (applied in order):
  1. Decode HTML entities (&amp; -> &, &gt; -> >, etc.)
  2. Normalize @mentions -> @user (preserve the fact that someone was mentioned,
     but remove the specific handle to reduce noise).
  3. Preserve URLs (needed for grounding audit -- we must verify generated URLs
     exist in retrieval context).
  4. Normalize whitespace (collapse runs, strip).
  5. Preserve emojis (important for brand voice and sentiment).
  6. Extract SpotifyCares agent initials from sign-offs (e.g. /AY, /CP, /MU).

All I/O is explicitly UTF-8.
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402

# Regex patterns
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")
_AGENT_SIGN_RE = re.compile(r"\s*/([A-Z]{2,3})\s*$")  # e.g. " /AY" at end


def clean_text(text: str) -> str:
    """Apply all text normalization steps. Returns cleaned text."""
    # 1. HTML entities
    text = html.unescape(text)

    # 2. Normalize @mentions -> @user
    text = _MENTION_RE.sub("@user", text)

    # 3. URLs are preserved as-is (needed for grounding audit)

    # 4. Normalize whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text


def extract_agent_initials(text: str) -> str | None:
    """
    Extract agent initials from a SpotifyCares sign-off.

    Examples:
        "Thanks for reaching out /AY"  ->  "AY"
        "We're here for you 😉 /CP"     ->  "CP"
        "Hello there"                   ->  None
    """
    match = _AGENT_SIGN_RE.search(text)
    return match.group(1) if match else None


def clean_threads(
    input_path: str | Path = config.SPOTIFY_THREADS_JSONL,
    output_path: str | Path | None = None,
) -> dict:
    """
    Clean all message texts in threads JSONL, in place or to a new file.

    Adds 'clean_text' and 'agent_initials' fields to each message.
    Returns summary stats.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path  # overwrite in place
    output_path = Path(output_path)

    threads: list[dict] = []
    total_messages = 0
    initials_found = 0

    with open(input_path, "r", encoding="utf-8") as fh:
        for line in fh:
            thread = json.loads(line)
            for msg in thread["messages"]:
                total_messages += 1
                msg["clean_text"] = clean_text(msg["text"])

                # Extract agent initials for outbound brand messages
                if not msg["inbound"]:
                    initials = extract_agent_initials(msg["text"])
                    msg["agent_initials"] = initials
                    if initials:
                        initials_found += 1
                else:
                    msg["agent_initials"] = None

            threads.append(thread)

    # Write output
    with open(output_path, "w", encoding="utf-8") as out:
        for thread in threads:
            out.write(json.dumps(thread, ensure_ascii=False) + "\n")

    summary = {
        "total_messages_cleaned": total_messages,
        "agent_initials_found": initials_found,
        "output_path": str(output_path),
    }
    print(f"[text_cleaner] Cleaned {total_messages:,} messages, "
          f"found {initials_found:,} agent initials")
    return summary


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = clean_threads()
    print(json.dumps(result, indent=2))
