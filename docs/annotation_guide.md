# Annotation Guide -- SpotifyCares AI Support Agent

## Overview

This guide describes how to label customer tweets for the SpotifyCares classification and escalation system. Every label must be consistent with the [taxonomy](taxonomy.md) and [escalation policy](escalation_policy.md).

## Label Schema

Each example receives the following labels:

```json
{
  "tweet_id": "119256",
  "thread_id": "119256",
  "text": "@SpotifyCares Please help! Spotify Premium skipping through songs constantly on android tablet & bluetooth speaker. Tried everything!",
  "clean_text": "@user @user Please help! Spotify Premium skipping through songs constantly on android tablet & bluetooth speaker. Tried everything!",
  "route": "support",
  "domain": "app_device",
  "risk_flag": "none",
  "escalation_decision": "auto_handle",
  "escalation_reason": "Standard app/device issue with likely retrieval evidence",
  "difficulty": "easy",
  "notes": "Could also be classified as playback -- classified as app_device because the customer mentions specific device/speaker setup"
}
```

### Field Definitions

| Field | Type | Values | Description |
|---|---|---|---|
| `route` | string | `support`, `feedback`, `abuse_spam` | What kind of message (see taxonomy Layer 1) |
| `domain` | string or null | `playback`, `account`, `billing`, `content`, `app_device`, `how_to`, `null` | What the issue is about (Layer 2, only if route=support) |
| `risk_flag` | string | `security`, `payment_dispute`, `legal`, `repeated_contact`, `unclear`, `none` | Special handling needed? (Layer 3) |
| `escalation_decision` | string | `auto_handle`, `escalate`, `clarify` | What should happen with this message |
| `escalation_reason` | string | free text | Why this decision was made -- must reference a specific policy rule |
| `difficulty` | string | `easy`, `medium`, `hard` | How confident you are in this label |
| `notes` | string or null | free text | Any ambiguity, alternative labels, or context needed |

## Labeling Procedure

1. **Read the full thread** (not just the individual tweet). Context matters for risk flags like `repeated_contact`.
2. **Assign Route first.** If not `support`, skip Domain.
3. **Assign Domain** based on the primary issue. See boundary rules in taxonomy.md.
4. **Assign Risk Flag** by checking all risk conditions. If multiple apply, use the highest-severity: `security` > `legal` > `payment_dispute` > `repeated_contact` > `unclear` > `none`.
5. **Determine Escalation Decision** by applying the escalation policy rules.
6. **Write the Escalation Reason** citing the specific policy rule that applies.
7. **Rate Difficulty:** `easy` = obvious classification; `medium` = required reading the full thread or thinking about boundaries; `hard` = genuinely ambiguous, reasonable annotators could disagree.
8. **Add Notes** for anything non-obvious.

## Common Pitfalls

- **Don't confuse frustration with abuse.** A frustrated customer ("this is ridiculous!!!") is still `support`, not `abuse_spam`.
- **Sarcasm is support.** "Love how my app crashes every 5 minutes!" -> `route: support`.
- **Thread length matters.** A 6-message thread with no resolution -> risk: `repeated_contact` even if the latest message doesn't mention it.
- **"DM us" responses don't change the label.** Label based on the customer's message, not the brand's response.
- **Multi-issue tweets:** Label the dominant issue. Note the secondary in `notes`.

## Quality Control

- After initial labeling, wait at least 2 days, then re-label 50 randomly-selected examples blind.
- Compare: track how many labels changed and which fields are most unstable.
- Report intra-annotator consistency in labeling_notes.md.
