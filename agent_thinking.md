# Agent Thinking Process — Hiver SDE Intern Assignment (Revised)

> This document captures the **full reasoning chain** behind every architectural and design decision. Updated with corrections from feedback and grounded in full-dataset audit results.

---

## 1. Problem Decomposition & Understanding

### 1.1 What is actually being asked?

Two meta-goals:

1. **Build a working AI support agent** — intent classification, reply drafting, escalation decision.
2. **Prove it works** — golden eval set, automated metrics, LLM-as-judge, failure analysis, misleading metrics section.

The assignment explicitly says: *"The proof is worth more than the system."* This is the core insight. The evaluators care more about **rigorous evaluation methodology** than a fancy model. The strongest narrative is: *"I built a constrained drafting agent that only auto-handles well-supported, low-risk cases; I prove where it works, where it abstains, and why the headline number can mislead."*

### 1.2 Implicit Requirements

- **"Messy real-world dataset"** — Data cleaning skills. Twitter noise: URLs, @mentions, hashtags, abbreviations, emojis, non-trivial thread reconstruction.
- **"Pick one brand"** — Scope control. Depth over breadth.
- **"Intents that you define from the data"** — No pre-defined taxonomy. Requires data exploration → clustering → manual refinement.
- **"Grounded in how that brand has historically resolved similar issues"** — RAG, but critically: these are **public Twitter responses**, not verified resolutions. Many threads end in "DM us" and the actual resolution is private. The retrieval corpus is "historical public support patterns," not a knowledge base.
- **"Auto-handle or escalate with a stated reason"** — Confidence-based routing with a written policy.
- **"What is misleading about my headline number?"** — Mandatory intellectual honesty section.
- **"Reproduce in under 15 minutes"** — No training from scratch. Pre-computed artifacts + cached outputs for the standard path.

---

## 2. Full Dataset Audit Results

### 2.1 Correcting the Initial Analysis

My initial analysis sampled only the first 300K rows and computed "multi-turn %" incorrectly (measured whether an outbound tweet had an `in_response_to_tweet_id`, which is ~99.8% by definition — almost every brand response is replying to something). This was misleading.

### 2.2 Actual Numbers (Full 2.8M-row Dataset)

| Metric | Value |
|---|---|
| Total dataset rows | 2,811,774 |
| SpotifyCares outbound tweets | 43,265 |
| Total tweets in SpotifyCares conversations | 88,445 |
| Reconstructed threads | 29,485 |
| Threads with ≥ 3 messages (real multi-turn) | 9,460 (32.1%) |
| Threads with ≥ 5 messages (rich multi-turn) | 3,788 (12.8%) |
| DM redirect rate | 30.8% |
| Responses with troubleshooting steps | 9.7% |
| Responses with links (KB articles) | 50.5% |
| Responses with diagnostic questions | 45.0% |
| Avg response length | 130 chars |
| Branching tweets (multiple responses) | 3,475 |
| Temporal range | Sep 2013 – Dec 2017 (99%+ in Oct-Nov 2017) |

### 2.3 Critical Implications

1. **67.9% of threads are just 2 messages** (customer → single brand response). These are NOT rich multi-turn conversations. My initial claim of "rich troubleshooting" was wrong.

2. **30.8% of responses contain DM redirects.** The actual resolution happened in private DMs, which we don't have. The retrieval corpus is **incomplete by design** — it contains public patterns (acknowledge, ask diagnostic questions, suggest basic steps, redirect to DM/links), not full resolutions.

3. **Only 9.7% contain explicit troubleshooting steps** (log out, restart, reinstall, etc.). The majority are diagnostic questions (45%) or link-to-KB responses (50.5%).

4. **The data is temporally concentrated** — 95%+ of tweets are from Oct-Nov 2017. This limits any "chronological split" to a narrow window, but we should still use temporal ordering for train/test separation.

5. **3,475 branching tweets** — non-trivial to handle. Must be preserved or deterministically resolved with the rule documented.

### 2.4 Revised Framing

The retrieval corpus should be framed as: **"historical public support patterns — how SpotifyCares publicly acknowledges, triages, and begins resolving issues on Twitter."** NOT: "a knowledge base of verified resolutions."

The agent's realistic role is: **draft an initial public response** that acknowledges the issue, asks the right diagnostic questions, provides basic troubleshooting if appropriate, and decides whether to escalate or redirect to private support. It is NOT a full resolution engine.

---

## 3. Brand Selection (Revised)

### 3.1 Full-Dataset Comparison

| Brand | Outbound Tweets (Full) | Notes |
|---|---|---|
| AmazonHelp | 169,840 | Largest but heavily generic/DM-redirect |
| AppleSupport | 106,860 | Mostly iOS complaint → "DM us" templates |
| Uber_Support | 56,270 | Worth investigating but likely ride-specific temporal issues |
| **SpotifyCares** | **43,265** | Rich diagnostic questioning pattern, links to KB |
| Delta | 42,253 | Flight-specific, heavily temporal |
| Tesco | 38,573 | UK-specific, diverse topics |

### 3.2 SpotifyCares Still Holds — But With Caveats

SpotifyCares remains a defensible choice because:
- **45% of responses ask diagnostic questions** — a learnable triage pattern
- **50.5% include links** — the agent learns to reference external resources
- **9.7% include explicit troubleshooting** — a minority, but these are the highest-value retrieval targets
- The topic space (music streaming) is relatively bounded

But I now acknowledge: most threads are short, and the "depth" of resolution in public Twitter data is limited. The agent is learning **triage and initial response patterns**, not complete technical support.

---

## 4. Architecture Design (Revised)

### 4.1 High-Level Pipeline (unchanged in structure)

```
Customer Tweet → Preprocessing → Classification (Route + Domain + Risk) → { Reply Drafting + Escalation Decision }
                                                                                      ↓
                                                                           Thread-aware RAG retrieval
                                                                           (training data only, never test threads)
```

### 4.2 Modular Design (decision stands)

The modular approach still wins because evaluation rigor requires independent measurement of each component. The key correction is that **the retrieval corpus must be strictly separated from the test set by thread**.

### 4.3 Hierarchical Intent Schema (revised from flat taxonomy)

The flat taxonomy had critical overlaps: `playback_issue`, `app_crash_bug`, and `device_compatibility` co-occur frequently. `account_access` conflates normal login with high-risk compromise. `positive_feedback` and `offensive_spam` are routing categories, not support intents.

**Revised Schema:**

```
Layer 1 — Route:     support | feedback | abuse/spam
Layer 2 — Domain:    playback | account | billing | content | app/device | how-to
Layer 3 — Risk Flag: security | payment-dispute | legal | repeated-contact | unclear | none
```

**Why this is better:**

1. **Route** separates messages that need a support response from those that don't (feedback/spam). This immediately simplifies downstream.
2. **Domain** captures *what the issue is about* without forcing false distinctions between overlapping failure modes. A message about "Spotify crashes when playing via Bluetooth" is `domain: app/device` — it doesn't need to be forced into either "app crash" or "device compatibility."
3. **Risk Flag** captures *whether the issue is safe to automate* — orthogonal to what it's about. A normal account login question (no risk) vs. "someone hacked my account" (security risk) are both `domain: account` but have different risk flags.

This makes the escalation decision much cleaner: classify what it is (Route + Domain), then separately decide whether it is safe to automate (Risk Flag).

### 4.4 Data Leakage Prevention (new — critical)

**The biggest technical risk is evaluation leakage.** This was correctly identified in feedback.

**Rules:**

1. **Split by entire conversation thread**, never by individual tweet. If tweet A and tweet B are in the same thread, they are in the same split.
2. **Prefer chronological split**: older threads → retrieval/training corpus; newer threads → dev set and test set.
3. **For every test tweet, exclude its real historical reply AND every message in its thread from retrieval.** Otherwise RAG can retrieve the answer being evaluated.
4. **The dev set and test set must not overlap.** Dev set for tuning (K, thresholds, prompts), test set for final numbers only.

**Proposed Split (based on temporal data — 95% is Oct-Nov 2017):**

Given the narrow temporal window, we split by date within Oct-Nov 2017:
- **Training/Retrieval corpus**: Oct 1–25, 2017 threads (~60%)
- **Development set**: Oct 26–31, 2017 threads (~15%)
- **Frozen test set**: Nov 1+, 2017 threads (~25%)

Alternative: random thread-level split with temporal ordering as a tiebreaker.

**Implementation detail:** At retrieval time, the system receives the test tweet's thread ID and filters it out of the search results. This is a hard constraint, not an optimization.

### 4.5 Reply Generation — Revised Framing

**What the RAG corpus actually contains:**
- Diagnostic question patterns ("What device are you on?")
- Basic troubleshooting suggestions ("Try logging out and back in")
- Link referrals to KB articles
- DM redirects ("Send us a DM with your account details")
- Empathetic acknowledgments ("We're sorry to hear that")

**What it does NOT contain:**
- Verified resolution steps
- Internal policies or procedures
- Account-specific information
- Complete technical documentation

**Therefore the agent should:**
- Draft an initial public response using these patterns
- Include a **safety check**: never invent policies, links, refunds, account actions, or request sensitive information publicly
- Default to "clarify or redirect to DM" when evidence is weak

### 4.6 Escalation — Written Policy (new, before tuning)

The feedback correctly insists: **define a written policy before tuning anything.**

**SpotifyCares Escalation Policy:**

| Condition | Action | Rationale |
|---|---|---|
| Risk flag: `security` / unauthorized access | **Always escalate** → private support | Account safety; cannot discuss security publicly |
| Risk flag: `legal` | **Always escalate** → private support | Legal exposure |
| Risk flag: `payment-dispute` / billing | **Escalate** → human review | Financial impact; no $ threshold since data doesn't provide values |
| Risk flag: `repeated-contact` (thread ≥ 5 back-and-forth unresolved) | **Escalate** → human review | Customer patience exhausted |
| Risk flag: `unclear` / low calibrated confidence | **Clarify first**, then escalate if still unclear | Avoid wrong auto-response |
| Route: `abuse/spam` | **Ignore / flag** | Not a support conversation |
| Route: `feedback` | **Acknowledge only** (auto-handle) | No action needed |
| Domain: `playback` / `how-to` / `app/device` with strong retrieval evidence | **Auto-handle** | Common issues with known patterns |
| Domain: `content` (missing songs/podcasts) | **Auto-handle** (acknowledge + check link) | Regional/licensing; limited agent ability |
| Domain: `account` without security flag | **Auto-handle** (standard login help) | Well-documented recovery steps |

This policy is the ground truth for escalation labeling in the golden set. It must be written BEFORE any labeling begins.

### 4.7 Confidence Calibration (addressing feedback)

The feedback correctly notes: **LLM confidence is not inherently calibrated.**

**Approach:**
1. Do NOT treat raw LLM probability/logprobs as calibrated confidence.
2. Instead, measure a **coverage–risk curve** on the development set:
   - At each confidence threshold T, compute: "the agent auto-handles X% of cases (coverage) with Y% error rate (risk)."
   - Select T to achieve a target error rate (e.g., < 10% on auto-handled cases).
   - Report the curve in the final report, not just a single threshold.
3. For the embedding-based classifier, use softmax temperature scaling on the dev set as a calibration step.

---

## 5. Evaluation Strategy (Revised)

### 5.1 Three Separate Sets (not two)

| Set | Size | Purpose | When Created | When Used |
|---|---|---|---|---|
| **Training labels** | 300–500 examples | Train TF-IDF/LR baseline, few-shot examples | Before any model work | During model development |
| **Development set** | 100–150 examples | Tune K, thresholds, prompts, escalation weights | Before model work | During model development |
| **Frozen test set** | 200 examples | Final reported results only | Before model work | Once, at the end |

**Critical:** The TF-IDF classifier trains on the training labels, NOT the test set. The test set is touched exactly once for final numbers.

### 5.2 No "Inter-Annotator Agreement" from Self-Consistency

The feedback rightly flags: you cannot claim inter-annotator agreement from a single annotator checking their own work. Options:
- Recruit one additional labeler for a 50-example subset. Report Cohen's Kappa.
- If solo: label 50 examples, wait a week, re-label blind. Report intra-annotator consistency.
- Be honest in the report: "single-annotator labels; consistency checked via re-labeling 50 examples after N days."

### 5.3 Reply Evaluation — Headline Metrics (revised)

The feedback correctly identifies that BLEU/ROUGE/BERTScore are weak headline metrics for reply generation (many valid replies exist for one tweet). These become **secondary diagnostics only**.

**Primary (Headline) Evaluation:**

1. **LLM-judge rubric (blinded, validated):**
   - Relevance (1-5): Does the reply address the customer's actual issue?
   - Actionability (1-5): Clear next step for the customer?
   - Brand voice (1-5): Matches SpotifyCares' warm, helpful tone?
   - Safety/Privacy (1-5): No invented policies, links, refunds, account actions, or PII requests in public?
   - Evidence-supportedness (1-5): Is each claim/recommendation supported by retrieved historical examples? *(renamed from "Accuracy" — we have no authoritative product knowledge source)*

2. **Pairwise comparison:** Retrieval-only baseline vs. RAG generation. For each test example, present both responses to the LLM-judge (position-randomized), ask which is better and why. Report win rate + tie rate.

3. **Grounding audit:** For a manual subset (30-50 examples), trace each recommendation in the generated reply back to a specific retrieved example. Report: % of recommendations that are grounded vs. hallucinated.

4. **Safety audit:** Scan all generated replies for:
   - Invented URLs (not in retrieval results)
   - Made-up refund/compensation offers
   - Requests for sensitive information (passwords, payment details) in public
   - Specific account actions the agent can't actually take
   - Policy claims not supported by historical data

**Secondary (Diagnostic) Metrics:**
- BLEU-4, ROUGE-L, BERTScore (reported but explicitly disclaimed)
- Response length distribution (within 280-character Twitter limit)
- Retrieval hit rate (% of test queries where at least one relevant result is retrieved)

### 5.4 "What Is Misleading About My Headline Number?" (expanded)

1. **Distribution bias**: Stratified eval set doesn't reflect real-world distribution. Easy common intents dominate production traffic.
2. **Template inflation**: BLEU/BERTScore reward matching SpotifyCares' formulaic patterns, not genuine understanding.
3. **LLM-judge circular reasoning**: If generation and judge use the same model family, judge may prefer outputs that "sound right" to that model.
4. **Surviving selection bias**: Only threads that stayed public are in the dataset. The hard cases went to DM. Our eval set is biased toward easier, publicly-resolvable issues.
5. **Temporal monoculture**: 95%+ of data is Oct-Nov 2017. Performance on 2024/2025 issues is unknown.
6. **Escalation precision is cheap**: Conservative escalation (only obvious cases) gives high precision but low recall — the dangerous direction.
7. **The agent never truly "resolves" anything**: It drafts a first response. Resolution requires DM follow-up, account access, internal tools. Our metrics measure draft quality, not resolution quality.

---

## 6. Revised Baseline Design

### 6.1 Trivial Baseline
- **Classification**: Always predict most-frequent route + domain.
- **Reply**: Fixed template: "Hi there! We'd love to help. Could you send us a DM with more details about the issue you're experiencing? /AI"
- **Escalation**: Never escalate (or always escalate — report both).

### 6.2 Simple (Non-LLM) Baseline
- **Classification**: TF-IDF + Logistic Regression trained on 300-500 labeled training examples. Zero-shot for the hierarchical risk flags (keyword-based).
- **Reply**: 1-NN retrieval — most similar historical response from training corpus (no LLM generation).
- **Escalation**: Transparent rule-based — keyword match for security/legal/billing + thread length > 4 for repeated-contact.

### 6.3 Full System
- **Classification**: LLM-based hierarchical classification (Route → Domain → Risk flag).
- **Reply**: Thread-aware RAG retrieval (from training data only) + LLM generation with safety checks.
- **Escalation**: Policy-based + coverage-risk calibrated confidence.

---

## 7. Revised Implementation Sequence (Scope-Realistic)

The feedback correctly identifies the original plan as over-scoped. Revised to a 6-step sequence:

### Step 1: Data Audit First
- ✅ Full-dataset extraction (done — 88K tweets, 29K threads)
- Reconstruct threads with quality metrics (graph completeness, dangling references)
- Measure: thread length distribution, DM-redirect rate, branch rate, temporal range
- Produce a **fixed split manifest** (thread IDs assigned to train/dev/test)
- Document: duplicate handling, branch resolution rule, encoding issues

### Step 2: Define Labels and Policy
- Write the hierarchical taxonomy (Route → Domain → Risk flag)
- Write the escalation policy (before any model work)
- Write the annotation guide (with examples for each category)
- Write explicit exclusions (what we chose NOT to build and why)
- Label training set (300-500 examples) + dev set (100-150) + frozen test set (200)

### Step 3: Build a Credible Non-LLM Baseline
- TF-IDF + Logistic Regression for classification (trained only on training labels)
- 1-NN historical response retrieval
- Transparent rule-based escalation
- Run on dev set, report metrics, use to calibrate expectations

### Step 4: Build the Minimal Full Agent
- Thread-aware retrieval from training data only (ChromaDB, exclude test threads)
- Concise draft generator (configurable LLM provider/model)
- Hard safety/length checks (280-char limit, no invented URLs, no PII requests)
- Clarification/DM fallback when retrieval evidence is weak

### Step 5: Evaluate and Ablate
- Compare: trivial baseline → simple baseline → retrieval-only → full RAG generation
- Ablate: K=3 vs K=5, with/without intent-filtered retrieval
- LLM-judge evaluation (validated against 50 human-scored examples)
- Safety audit, grounding audit
- Failure analysis: top 5 failure modes with real examples and hypotheses
- Coverage-risk curve for escalation

### Step 6: Package Proof
- One command to reproduce: use supplied processed data + cached outputs
- No large downloads, no embedding recomputation, no paid API calls for standard reproduction path
- README with clear reproduction instructions (< 15 min)
- Report (max 6 pages) with all required sections

---

## 8. Technical Corrections

### 8.1 Character Limit
Twitter's character limit is **280 characters** (since November 2017). This is explicitly set in code and documented.

### 8.2 Branching Threads
3,475 tweets have multiple responses. Rule: **preserve all branches** in the data. For retrieval, include all response variants. For evaluation, if a test tweet has multiple valid responses, treat each as an acceptable reference. Document this decision.

### 8.3 Configurable LLM Provider
Do not commit the report to named models. Use a configurable `LLM_PROVIDER` / `LLM_MODEL` setting with documented defaults. The evaluation should be reproducible with any capable LLM.

### 8.4 Encoding
All files read/written as UTF-8 explicitly. The garbled characters (`â€"`, `â†'`) from the terminal are fixed.

### 8.5 Results Framing
The "Expected Results" table contains **hypotheses**, not evidence. The report will clearly distinguish between targets and observed results. The final report will say "we observed X" not "we expect X."

---

## 9. Revised Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Data leakage** (test thread in retrieval) | High if not careful | Critical | Thread-level split, retrieval filter by thread_id, automated leak check |
| **DM-redirect dominance** (30.8% of responses) | Certain | High | Acknowledge in framing; measure and report separately |
| **Temporal shift** (Oct-Nov 2017 only) | Certain | Medium | Document; do NOT claim generalization |
| **Calibration failure** (confidence ≠ quality) | Medium | High | Coverage-risk curve on dev set; never trust raw logprobs |
| **Over-scoping** | High | High | Follow the 6-step sequence strictly; cut features, not quality |
| **Single annotator bias** | Certain | Medium | Document honestly; intra-annotator re-labeling for consistency |
| **API cost/availability** | Medium | Medium | Configurable provider; cached outputs for reproduction |

---

## 10. What I'm Deliberately NOT Building (and why)

1. **Full resolution engine** — The public data doesn't contain resolutions. We build a triage/draft agent.
2. **Fine-tuned models** — Hard to reproduce, training time exceeds scope, marginal improvement over good prompting + RAG.
3. **Multi-brand support** — Depth over breadth. One brand, done well.
4. **Production deployment (Docker, API server)** — Not asked for. A CLI pipeline suffices.
5. **Official Spotify documentation integration** — Would strengthen "evidence-supportedness" but is outside the dataset scope. Noted as "what I'd do with one more week."
6. **Sentiment analysis model training** — Use off-the-shelf (VADER) for the risk signal. Not the core contribution.
7. **"Billing dispute > $X" threshold** — The data doesn't reliably provide transaction values. All billing disputes escalate per policy.
