# Hierarchical Classification Taxonomy

This document defines the classification schema used by the SpotifyCares AI Support Agent.
It is written **before any model work** and serves as the ground truth for annotation.

---

## Schema Overview

Classification is three-layered. Each incoming message receives **one label per layer**:

```
Layer 1 — Route:      What kind of message is this?
Layer 2 — Domain:     What topic is the support request about?  (only if Route = support)
Layer 3 — Risk Flag:  Does this require special handling?       (always assigned)
```

---

## Layer 1 — Route

| Route | Definition | Examples |
|---|---|---|
| `support` | Customer is reporting a problem, asking a question, or requesting help. Expects a resolution or next step. | "My Spotify keeps crashing", "How do I cancel my subscription?", "Songs won't play on my Bluetooth speaker" |
| `feedback` | Customer is providing feedback, praise, complaint-without-action-request, or general commentary. No specific resolution expected. | "Thanks for fixing it!", "I love the new Discover Weekly", "Spotify is the best app ever" |
| `abuse_spam` | Message is abusive, spam, completely off-topic, or not a genuine support interaction. | Profanity-only messages, promotional spam, unrelated political commentary |

**Boundary rules:**
- A frustrated complaint WITH an implied request for help → `support` (not `feedback`).
- "Your app sucks" with no further detail → `feedback`. "Your app sucks, it keeps crashing" → `support`.
- Sarcastic praise ("Love how my app crashes every 5 minutes!") → `support`.

---

## Layer 2 — Domain (only when Route = `support`)

| Domain | Definition | Examples |
|---|---|---|
| `playback` | Issues with playing music: skipping, pausing, buffering, no sound, wrong song plays, shuffle/repeat issues. | "Songs keep skipping", "No sound on my speakers", "Shuffle isn't random enough" |
| `account` | Login, logout, password reset, account recovery, profile changes, connected accounts (Facebook, etc.). | "Can't log in", "Reset my password", "My account got hacked" |
| `billing` | Payment failures, subscription plan changes, refund requests, pricing questions, free trial issues. | "I was charged twice", "How to switch to family plan?", "Cancel my subscription" |
| `content` | Missing songs, albums, podcasts; region-locked content; content quality; playlist issues. | "This album isn't available", "Podcast episodes missing", "My playlist was deleted" |
| `app_device` | App crashes, freezes, installation issues, device-specific problems, OS compatibility, Bluetooth/speaker issues, offline mode. | "App crashes on my iPhone", "Can't download for offline", "Bluetooth speaker keeps disconnecting" |
| `how_to` | How-to questions, feature discovery, general usage help. | "How do I make a collaborative playlist?", "Where are my listening stats?", "How to use Spotify Connect?" |

**Boundary rules:**
- "App crashes when playing via Bluetooth" → `app_device` (the issue is the crash/device interaction, not the music itself).
- "Song stops halfway through" → `playback` (even if it might be a device issue — classify by the symptom the customer reports).
- "Can't log in and I was charged" → Use the **primary** complaint. If ambiguous, label the one the customer emphasizes more. If truly equal, prefer the higher-risk domain (`billing` > `account` > others).
- Multi-issue messages: Label by the **dominant** issue. Note the secondary issue in annotation comments.

---

## Layer 3 — Risk Flag (always assigned)

| Risk Flag | Definition | Escalation Implication | Examples |
|---|---|---|---|
| `security` | Account compromise, unauthorized access, stolen credentials, suspicious activity. | **Always escalate** → private support | "Someone hacked my account", "I see playlists I didn't create", "Unauthorized login from another country" |
| `payment_dispute` | Disputed charges, double billing, unauthorized payment, refund demands. | **Escalate** → human review | "I was charged $9.99 but I cancelled", "Unauthorized charge on my card", "I want a refund" |
| `legal` | Mentions of lawyers, lawsuits, regulatory complaints, GDPR/privacy requests. | **Always escalate** → private support | "I'll sue", "GDPR data deletion request", "I'm contacting the BBB" |
| `repeated_contact` | Customer indicates they've contacted support multiple times without resolution. Detected from thread length ≥ 5 or explicit mentions. | **Escalate** → human review | "This is the third time I'm asking", "I've been trying for weeks" (also detected via thread metadata) |
| `unclear` | Intent is ambiguous, message is too vague to classify confidently, or multiple conflicting signals. | **Clarify first**, then escalate | "Help", "Something's wrong", incomprehensible messages |
| `none` | No special risk. Standard support interaction. | Follow domain-based auto-handle rules | Most routine support questions |

**Boundary rules:**
- "I forgot my password" → `account` + risk: `none` (standard recovery).
- "Someone changed my password" → `account` + risk: `security`.
- "I want my money back" → `billing` + risk: `payment_dispute`.
- "Why did my price go up?" → `billing` + risk: `none` (inquiry, not dispute).
- Thread with 6 back-and-forth messages → risk: `repeated_contact` (from thread metadata, regardless of message content).

---

## Label Combination Examples

| Customer Message | Route | Domain | Risk Flag | Reasoning |
|---|---|---|---|---|
| "Spotify keeps crashing on my Samsung Galaxy" | support | app_device | none | Standard device issue |
| "SOMEONE IS USING MY ACCOUNT" | support | account | security | Account compromise → always escalate |
| "How do I make a playlist?" | support | how_to | none | Simple how-to |
| "I was charged twice this month, fix it NOW" | support | billing | payment_dispute | Financial dispute |
| "Thanks, that fixed it! 😊" | feedback | — | none | Positive feedback, no action needed |
| "I've contacted you 4 times about this and nobody helped" | support | *(from context)* | repeated_contact | Repeated unresolved contact |
| "lol spotify sux" | feedback | — | none | Non-actionable negative feedback |
| "f*** you spotify" | abuse_spam | — | none | Abusive, no support intent |
| "Help" | support | — | unclear | Too vague to classify domain |
