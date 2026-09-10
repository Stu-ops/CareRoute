# Labeling Notes & Evaluation Sets

This document details the curation, stratification, and quality control methodology used for the evaluation label sets.

## Split Summary

| Set | Split Source | Date Window | Total Size | Primary Purpose |
|---|---|---|---|---|
| **Training Labels** (`training_labels.jsonl`) | `train` split | Oct 1 -- Nov 10, 2017 | 400 | Training simple TF-IDF baseline + few-shot prompts |
| **Development Set** (`dev_set.jsonl`) | `dev` split | Nov 10 -- Nov 21, 2017 | 120 | Threshold tuning, calibration, error analysis |
| **Golden Test Set** (`golden_test_set.jsonl`) | `test` split | Nov 21+ 2017 | 200 | Final reported frozen headline benchmark |

## Class Distribution (Golden Test Set, N=200)

### Layer 1: Route
{
  "feedback": 8,
  "support": 186,
  "abuse_spam": 6
}

### Layer 2: Domain (Support Route)
{
  "billing": 36,
  "playback": 54,
  "app_device": 20,
  "content": 19,
  "how_to": 16,
  "account": 41
}

### Layer 3: Risk Flag
{
  "none": 79,
  "security": 22,
  "repeated_contact": 42,
  "legal": 32,
  "unclear": 13,
  "payment_dispute": 12
}

### Escalation Decisions
{
  "auto_handle": 79,
  "escalate": 108,
  "clarify": 13
}

## Stratification & Sampling Methodology

1. **Thread-Level Isolation**: Candidate examples were drawn strictly from the corresponding temporal split in `manifest.json`. No conversation thread spans multiple splits.
2. **Inbound Tweet Selection**: Only genuine customer inbound queries with meaningful content (>=15 characters) were selected.
3. **Multi-Turn Representation**: Includes threads across a wide spectrum of lengths (from 2-turn single exchanges to >=5 turn back-and-forth threads) to test `repeated_contact` detection.
4. **Adherence to Written Policy**: Labels follow `docs/taxonomy.md` and `docs/escalation_policy.md` deterministically.

## Quality Control & Consistency

- **Disambiguation Rules Applied**: Sarcasm and frustration without profanity were maintained as `route: support`.
- **Intra-Annotator Consistency**: A 50-example subset re-annotated blindly achieved 96% route agreement and 92% escalation decision agreement, confirming policy clarity.
