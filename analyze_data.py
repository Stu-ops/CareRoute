import csv
import collections

brands_of_interest = ['SpotifyCares', 'AppleSupport', 'AmazonHelp', 'Tesco', 'Delta']
stats = {b: {'outbound': 0, 'multi_turn': 0, 'lens': [], 'inbound_related': 0} for b in brands_of_interest}

total = 0
inbound_count = 0
outbound_count = 0

with open(r'c:\Coding\New folder\Dataset_1\twcs\twcs.csv', 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += 1
        author = row.get('author_id', '')
        text = row.get('text', '')
        in_response = row.get('in_response_to_tweet_id', '')
        inbound = row.get('inbound') == 'True'

        if inbound:
            inbound_count += 1
        else:
            outbound_count += 1

        if author in stats:
            stats[author]['outbound'] += 1
            stats[author]['lens'].append(len(text))
            if in_response:
                stats[author]['multi_turn'] += 1

        if total > 300000:
            break

print(f"Total rows sampled: {total}")
print(f"Inbound: {inbound_count}, Outbound: {outbound_count}")
print()

for brand in brands_of_interest:
    s = stats[brand]
    avg = sum(s['lens']) / max(len(s['lens']), 1)
    print(f"{brand}:")
    print(f"  Outbound tweets: {s['outbound']}")
    print(f"  Multi-turn replies: {s['multi_turn']}")
    print(f"  Avg response length: {avg:.0f} chars")
    print()
