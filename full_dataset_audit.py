"""
Full-dataset audit for SpotifyCares on TWCS.
Measures: total tweets, thread reconstruction quality, thread lengths,
DM-redirect rate, branch rate, temporal range.
"""
import csv
import collections
import re
from datetime import datetime

CSV_PATH = r"c:\Coding\New folder\Dataset_1\twcs\twcs.csv"

# Pass 1: Collect ALL tweets involving SpotifyCares
# A tweet "involves" SpotifyCares if the author is SpotifyCares
# OR if it is an inbound tweet that SpotifyCares responded to.

print("=== Pass 1: Scanning full dataset ===")
all_tweets = {}          # tweet_id -> row dict
spotify_tweet_ids = set()
total_rows = 0
brand_counts = collections.Counter()

with open(CSV_PATH, "r", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f)
    for row in reader:
        total_rows += 1
        tid = row.get("tweet_id", "").strip()
        author = row.get("author_id", "").strip()
        inbound = row.get("inbound", "").strip() == "True"
        text = row.get("text", "")
        response_ids = row.get("response_tweet_id", "").strip()
        in_response_to = row.get("in_response_to_tweet_id", "").strip()
        created = row.get("created_at", "").strip()

        if not inbound:
            brand_counts[author] += 1

        if author == "SpotifyCares":
            spotify_tweet_ids.add(tid)
            all_tweets[tid] = {
                "tweet_id": tid,
                "author_id": author,
                "inbound": inbound,
                "text": text,
                "response_tweet_ids": [r.strip() for r in response_ids.split(",") if r.strip()],
                "in_response_to_tweet_id": in_response_to,
                "created_at": created,
            }
            # Also collect the tweet(s) SpotifyCares was responding to
            if in_response_to:
                for ref_id in in_response_to.split(","):
                    spotify_tweet_ids.add(ref_id.strip())
            # And tweets that responded to SpotifyCares
            for resp_id in [r.strip() for r in response_ids.split(",") if r.strip()]:
                spotify_tweet_ids.add(resp_id)

        if total_rows % 500000 == 0:
            print(f"  ...processed {total_rows:,} rows")

print(f"\nTotal rows in dataset: {total_rows:,}")
print(f"SpotifyCares outbound tweets: {len([t for t in all_tweets.values() if not t['inbound']])}")
print(f"Tweet IDs involving SpotifyCares (direct): {len(spotify_tweet_ids)}")

# Pass 2: Collect all referenced tweets (customer messages that SpotifyCares replied to)
print("\n=== Pass 2: Collecting customer tweets in SpotifyCares threads ===")
spotify_conversations = {}  # same dict but now includes customer tweets

with open(CSV_PATH, "r", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tid = row.get("tweet_id", "").strip()
        if tid in spotify_tweet_ids and tid not in all_tweets:
            author = row.get("author_id", "").strip()
            inbound = row.get("inbound", "").strip() == "True"
            text = row.get("text", "")
            response_ids = row.get("response_tweet_id", "").strip()
            in_response_to = row.get("in_response_to_tweet_id", "").strip()
            created = row.get("created_at", "").strip()
            all_tweets[tid] = {
                "tweet_id": tid,
                "author_id": author,
                "inbound": inbound,
                "text": text,
                "response_tweet_ids": [r.strip() for r in response_ids.split(",") if r.strip()],
                "in_response_to_tweet_id": in_response_to,
                "created_at": created,
            }

print(f"Total tweets in SpotifyCares conversations: {len(all_tweets)}")

# Thread reconstruction
print("\n=== Thread Reconstruction ===")

# Build adjacency
children = collections.defaultdict(list)  # parent_id -> [child_ids]
for tid, tweet in all_tweets.items():
    parent = tweet["in_response_to_tweet_id"]
    if parent and parent in all_tweets:
        children[parent].append(tid)

# Find roots (tweets with no in_response_to, or whose parent is not in our set)
roots = []
for tid, tweet in all_tweets.items():
    parent = tweet["in_response_to_tweet_id"]
    if not parent or parent not in all_tweets:
        roots.append(tid)

print(f"Root tweets (conversation starters): {len(roots)}")

# BFS to build threads
threads = []
visited = set()

for root in roots:
    if root in visited:
        continue
    thread = []
    queue = [root]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        if current in all_tweets:
            thread.append(all_tweets[current])
        for child in children.get(current, []):
            if child not in visited:
                queue.append(child)
    if thread:
        threads.append(thread)

print(f"Total reconstructed threads: {len(threads)}")

# Thread statistics
thread_lengths = [len(t) for t in threads]
print(f"\nThread length distribution:")
length_counts = collections.Counter(thread_lengths)
for length in sorted(length_counts.keys()):
    if length <= 15 or length_counts[length] > 5:
        print(f"  Length {length}: {length_counts[length]} threads")

# Multi-turn analysis (threads with >= 3 messages, i.e. at least one back-and-forth)
multi_turn_threads = [t for t in threads if len(t) >= 3]
rich_threads = [t for t in threads if len(t) >= 5]
print(f"\nThreads with >= 3 messages (real multi-turn): {len(multi_turn_threads)} ({100*len(multi_turn_threads)/max(len(threads),1):.1f}%)")
print(f"Threads with >= 5 messages (rich multi-turn): {len(rich_threads)} ({100*len(rich_threads)/max(len(threads),1):.1f}%)")

# DM redirect analysis
dm_patterns = re.compile(r"\b(DM|direct message|send us a (dm|message)|slide into|private message)\b", re.IGNORECASE)
dm_redirect_count = 0
total_outbound = 0
for thread in threads:
    for tweet in thread:
        if not tweet["inbound"] and tweet["author_id"] == "SpotifyCares":
            total_outbound += 1
            if dm_patterns.search(tweet["text"]):
                dm_redirect_count += 1

print(f"\nDM redirect analysis:")
print(f"  Total SpotifyCares outbound tweets: {total_outbound}")
print(f"  Tweets containing DM redirect: {dm_redirect_count} ({100*dm_redirect_count/max(total_outbound,1):.1f}%)")

# Branch analysis
branch_count = 0
for tid, tweet in all_tweets.items():
    if len(tweet["response_tweet_ids"]) > 1:
        branch_count += 1
print(f"\nBranching tweets (multiple responses): {branch_count}")

# Temporal range
dates = []
for tweet in all_tweets.values():
    try:
        dt = datetime.strptime(tweet["created_at"], "%a %b %d %H:%M:%S %z %Y")
        dates.append(dt)
    except:
        pass

if dates:
    dates.sort()
    print(f"\nTemporal range:")
    print(f"  Earliest: {dates[0]}")
    print(f"  Latest: {dates[-1]}")
    # Rough monthly distribution
    months = collections.Counter()
    for d in dates:
        months[d.strftime("%Y-%m")] += 1
    print(f"  Monthly distribution:")
    for month in sorted(months.keys()):
        print(f"    {month}: {months[month]} tweets")

# Content analysis: what does SpotifyCares actually say?
print("\n=== SpotifyCares Response Pattern Analysis ===")
troubleshoot_patterns = re.compile(r"(log out|restart|reinstall|clear cache|update|uninstall|try)", re.IGNORECASE)
link_patterns = re.compile(r"https?://", re.IGNORECASE)
question_patterns = re.compile(r"\?")

troubleshoot_count = 0
link_count = 0
question_count = 0
response_lengths = []

for tweet in all_tweets.values():
    if not tweet["inbound"] and tweet["author_id"] == "SpotifyCares":
        text = tweet["text"]
        response_lengths.append(len(text))
        if troubleshoot_patterns.search(text):
            troubleshoot_count += 1
        if link_patterns.search(text):
            link_count += 1
        if question_patterns.search(text):
            question_count += 1

print(f"  Responses with troubleshooting steps: {troubleshoot_count} ({100*troubleshoot_count/max(total_outbound,1):.1f}%)")
print(f"  Responses with links: {link_count} ({100*link_count/max(total_outbound,1):.1f}%)")
print(f"  Responses with questions (diagnostic): {question_count} ({100*question_count/max(total_outbound,1):.1f}%)")
print(f"  Avg response length: {sum(response_lengths)/max(len(response_lengths),1):.0f} chars")
print(f"  Median response length: {sorted(response_lengths)[len(response_lengths)//2] if response_lengths else 0} chars")

# Top 20 brands for comparison
print(f"\n=== Top 20 Brands (Full Dataset) ===")
for brand, count in brand_counts.most_common(20):
    print(f"  {brand}: {count:,}")

print(f"\n=== DONE ===")
