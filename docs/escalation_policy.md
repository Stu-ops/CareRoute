# Escalation Policy — SpotifyCares AI Support Agent

This policy is the **ground truth** for escalation labeling.
It is defined **before any model work** and must not be changed based on model output.

---

## Principles

1. **Safety over efficiency**: It is worse to auto-handle a risky case than to unnecessarily escalate a safe one.
2. **Transparency**: Every escalation decision must have a stated reason traceable to a policy rule.
3. **Private for sensitive**: Anything involving account security, payment disputes, or legal matters must move to private support (DM or phone).
4. **Clarify before guessing**: When intent is unclear, ask a clarifying question rather than making assumptions.

---

## Policy Rules

### Hard Escalations (always escalate, no exceptions)

| Condition | Action | Reason |
|---|---|---|
| Risk flag: `security` | Escalate to private support | Account safety -- cannot discuss security details publicly |
| Risk flag: `legal` | Escalate to private support | Legal exposure -- must involve internal team |
| Risk flag: `payment_dispute` | Escalate to human review | Financial impact -- agent cannot verify charges or issue refunds |

### Soft Escalations (escalate based on context)

| Condition | Action | Reason |
|---|---|---|
| Risk flag: `repeated_contact` (thread >= 5 unresolved exchanges) | Escalate to human review | Customer patience exhausted -- needs personalized attention |
| Risk flag: `unclear` OR classifier confidence below calibrated threshold | Send clarifying question first; escalate if still unclear after one round | Avoid incorrect auto-response |
| Multiple conflicting risk signals | Escalate to human review | Complex case requiring judgment |

### Auto-Handle (safe to automate)

| Condition | Action | Reason |
|---|---|---|
| Route: `feedback` | Acknowledge only (thank customer) | No action needed |
| Route: `abuse_spam` | Ignore / flag for review | Not a support conversation |
| Domain: `playback` + risk: `none` + strong retrieval evidence | Draft response from historical patterns | Common issue with known triage patterns |
| Domain: `app_device` + risk: `none` + strong retrieval evidence | Draft response from historical patterns | Common issue with known triage patterns |
| Domain: `how_to` + risk: `none` + strong retrieval evidence | Draft response from historical patterns | Well-documented usage questions |
| Domain: `content` + risk: `none` | Acknowledge + suggest checking availability / link | Limited agent ability -- content is licensing-dependent |
| Domain: `account` + risk: `none` | Draft standard recovery steps | Well-documented account recovery flows |
| Domain: `billing` + risk: `none` (inquiry, not dispute) | Provide general plan/pricing info | Non-sensitive billing inquiry |

### Fallback Rule

If no specific rule matches, or if retrieval evidence is weak (low similarity scores in top-K results):
- **Default to clarification/DM redirect**: "Could you send us a DM with more details so we can look into this further?"
- This is the safe default -- it mirrors what real SpotifyCares agents do 30.8% of the time.

---

## Confidence Calibration

- Raw LLM confidence scores are NOT inherently calibrated.
- The escalation confidence threshold is tuned on the **development set** (never the test set).
- The threshold is selected by analyzing a **coverage-risk curve**:
  - At each threshold T: "the agent auto-handles X% of cases (coverage) with Y% error rate (risk)."
  - We select T to achieve an acceptable risk level (target: < 10% error on auto-handled cases).
- The curve is reported in the final results.

---

## What the Agent Cannot Do

The agent drafts public Twitter responses only. It **cannot**:
- Access customer accounts
- Issue refunds or credits
- Reset passwords
- Verify payment information
- Make policy exceptions
- Access internal tools or databases

Any response that implies these capabilities is a **safety violation**.
