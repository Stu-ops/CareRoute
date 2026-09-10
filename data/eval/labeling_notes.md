# Labeling Notes & Evaluation Sets (Audited)

This document details the curation, stratification, and quality control methodology used for the evaluation label sets.

## Split Summary

| Set | Split Source | Date Window | Total Size | Primary Purpose |
|---|---|---|---|---|
| **Training Labels** (`training_labels.jsonl`) | `train` split | Oct 1 -- Nov 10, 2017 | 400 | Training simple TF-IDF baseline + few-shot prompts |
| **Development Set** (`dev_set.jsonl`) | `dev` split | Nov 10 -- Nov 21, 2017 | 120 | Threshold tuning, calibration, error analysis |
| **Golden Test Set** (`golden_test_set.jsonl`) | `test` split | Nov 21+ 2017 | 200 | Final reported frozen headline benchmark |

## Class Distribution (Golden Test Set, N=200)

### Layer 1: Route
```json
{
  "support": 190,
  "feedback": 7,
  "abuse_spam": 3
}
```

### Layer 2: Domain (Support Route)
```json
{
  "playback": 66,
  "billing": 45,
  "app_device": 27,
  "content": 20,
  "how_to": 17,
  "account": 15
}
```

### Layer 3: Risk Flag
```json
{
  "none": 110,
  "security": 11,
  "repeated_contact": 56,
  "unclear": 12,
  "payment_dispute": 11
}
```

### Escalation Decisions
```json
{
  "auto_handle": 110,
  "escalate": 78,
  "clarify": 12
}
```

## Quality Control & Systematic Audit Changelog

During comprehensive codebase review, systematic keyword false-positives were audited and corrected:
1. **Word-Boundary Isolation on Legal Terms**: Previously, substring matching for `sue` triggered on words like `issue` (e.g. "playlist issue", "offline issue", "billing issue"), mislabeling technical support queries as `legal` risk. Replaced with word-bounded `\bsue\b` and explicit legal terminology (`lawyer`, `lawsuit`, `subpoena`, `gdpr`).
2. **Account Security Disambiguation**: Previously, substring matching for `stranger` triggered on mentions of `"Stranger Things Mode"` setting or themed playlists, mislabeling content and device issues as `security`. Replaced with explicit account compromise patterns (`hack`, `unauthorized access`, `unauthorized login`, `account takeover`).
3. **Downloads & Offline Routing**: Mapped downloads, offline sync, and storage issues to `app_device` rather than `how_to`.
4. **Billing vs Payment Dispute**: Standard billing queries (student discount verification, subscription renewal) are classified under `domain: billing` with `risk_flag: none` (auto-handled with guidance to `spotify.com/account`), while double charges and unauthorized transactions escalate as `risk_flag: payment_dispute`.
