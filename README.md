# SpotifyCares AI Support Agent

> A constrained triage and drafting agent for SpotifyCares (Spotify's Twitter support) that classifies incoming messages, drafts grounded responses using historical public support patterns, and decides whether to auto-handle or escalate.

**Core narrative:** I built an agent that only auto-handles well-supported, low-risk cases. I prove where it works, where it abstains, and why the headline number can mislead.

---

## Quick Reproduction (< 15 minutes)

```bash
# 1. Clone and install
git clone <repo-url>
cd hiver-spotify-agent
pip install -r requirements.txt

# 2. Run with cached outputs (no API key needed)
python run_reproduction.py --cached

# 3. Or run with live API calls
export LLM_API_KEY="your-api-key"
export LLM_PROVIDER="openai"           # or groq, anthropic
export LLM_MODEL="gpt-4o-mini"
python run_reproduction.py --live
```

---

## Problem Framing

### What "good" means for SpotifyCares

A good agent for SpotifyCares must:
1. **Correctly triage** incoming messages (route, domain, risk level)
2. **Draft helpful initial responses** grounded in how SpotifyCares has historically handled similar issues publicly
3. **Know when to abstain** -- escalate to humans for security, billing disputes, legal, or when confidence is low
4. **Never fabricate** -- no invented URLs, refund offers, account actions, or PII requests

### What we chose NOT to build

- **A full resolution engine:** 30.8% of SpotifyCares responses redirect to DM. The actual resolution happens privately. Our agent learns *triage and initial response patterns*, not complete technical support.
- **A fine-tuned model:** Marginal improvement over well-prompted RAG, hard to reproduce, and out of scope for a take-home.
- **Multi-brand support:** Depth over breadth.
- **Official Spotify docs integration:** Would strengthen grounding, but outside dataset scope.

### Data Reality

| Metric | Value |
|---|---|
| Total TWCS dataset | 2,811,774 tweets |
| SpotifyCares conversations | 88,445 tweets in 29,485 threads |
| Real multi-turn threads (>= 3 msgs) | 9,460 (32.1%) |
| DM redirect rate | 30.8% |
| Responses with troubleshooting steps | 9.7% |
| Responses with links | 50.5% |
| Responses with diagnostic questions | 45.0% |
| Data temporal range | 95%+ in Oct-Nov 2017 |

---

## Architecture

```
Customer Tweet --> Preprocessing --> Hierarchical Classifier --> RAG Retrieval --> Reply Generator --> Safety Checks
                                    (Route/Domain/Risk)        (Training data)    (LLM + grounding)   (280 char, URLs, PII)
                                          |                                                                |
                                          v                                                                v
                                    Escalation Router                                             Final Response
                                    (Written policy +                                          (auto-handle OR
                                     calibrated threshold)                                      escalate + reason)
```

### Key Design Decisions

1. **Hierarchical classification**: Route (support/feedback/spam) -> Domain (playback/account/billing/...) -> Risk flag (security/legal/payment/none)
2. **Thread-level chronological split**: 17,256 train / 4,984 dev / 7,245 test threads. Zero leakage verified.
3. **Thread-aware RAG**: Retrieved results exclude the query's own thread to prevent leakage.
4. **Written escalation policy**: Defined before any model work. Security/legal/billing always escalate.
5. **Safety checks**: Post-generation scan for invented URLs, PII requests, fabricated policies.

---

## Results

> **IMPORTANT:** Numbers in this section are observed results from the frozen test set, evaluated exactly once.

*(To be filled after running evaluation)*

### Classification

| Metric | Trivial | Simple | Full Agent |
|---|---|---|---|
| Route Accuracy | | | |
| Domain Macro-F1 | | | |
| Risk Macro-F1 | | | |

### Reply Quality (LLM-Judge, 1-5 scale)

| Dimension | Trivial | Simple | Full Agent |
|---|---|---|---|
| Relevance | | | |
| Actionability | | | |
| Brand Voice | | | |
| Safety/Privacy | | | |
| Evidence-Supportedness | | | |

### Escalation

| Metric | Trivial | Simple | Full Agent |
|---|---|---|---|
| Precision | | | |
| Recall | | | |
| F1 | | | |
| FNR (missed escalations) | | | |

### Safety Audit

| System | Violation Rate | Target |
|---|---|---|
| Trivial | | < 2% |
| Simple | | < 2% |
| Full Agent | | < 2% |

---

## Failure Analysis

### Top 5 Failure Modes

*(To be filled after evaluation with real examples and hypotheses)*

1. **Multi-issue messages:** Customer reports two problems -- classifier picks one, misses the other.
2. **Sarcasm misclassification:** Sarcastic complaints routed as feedback instead of support.
3. **DM-redirect overuse:** Agent defaults to "send us a DM" when retrieval evidence is weak, even for simple issues with known answers.
4. **Temporal mismatch:** Training patterns from Oct 2017 may not match Nov 2017 issues if new features/bugs emerged.
5. **Edge-case risk detection:** Subtle security concerns ("someone else is using my account to listen to music") not caught by keyword patterns.

---

## What Is Misleading About My Headline Number?

1. **Surviving selection bias:** Only threads that stayed public are in the dataset. Hard cases went to DM. Our eval set is biased toward easier, publicly-resolvable issues.

2. **Template inflation:** SpotifyCares has formulaic response patterns. High BLEU/BERTScore may reward template-matching, not genuine understanding.

3. **Easy examples dominate:** Many customer messages have obvious intents ("my Spotify keeps crashing"). The headline accuracy is inflated by these easy cases.

4. **LLM-judge circular reasoning:** If generation and judge use models from the same family, the judge may prefer outputs that "sound right" to that model.

5. **Temporal monoculture:** 95%+ of data is from Oct-Nov 2017. Performance on current issues is unknown.

6. **The agent never truly resolves anything:** It drafts a first response. Resolution requires DM follow-up, account access, and internal tools. Our metrics measure *draft quality*, not *resolution quality*.

7. **Escalation precision is cheap:** Conservative escalation (only obvious cases) gives high precision but low recall -- the dangerous direction.

---

## What I'd Do With One More Week

1. **Integrate official Spotify help documentation** as a versioned knowledge source, enabling actual "accuracy" evaluation instead of just "evidence-supportedness."
2. **Fine-tune an embedding model** on SpotifyCares conversation pairs for better retrieval quality.
3. **Build a Streamlit demo** for interactive exploration of the agent's behavior.
4. **Add a second annotator** for a 50-example subset to report real inter-annotator agreement.
5. **Test on a held-out brand** (e.g., AppleSupport) to assess transfer learning potential.
6. **Implement prompt optimization** -- systematically vary prompt templates and measure impact on judge scores.

---

## Golden Evaluation Set

- **200 hand-labeled examples** from the frozen test split (Nov 21+ threads)
- Stratified by route/domain/risk + adversarial slice
- Labeled following `docs/annotation_guide.md`
- Sampling and labeling methodology documented in `data/eval/labeling_notes.md`

---

## Project Structure

```
hiver-spotify-agent/
├── README.md                    # This file (report + reproduction guide)
├── requirements.txt             # Dependencies
├── config.py                    # Configurable LLM/model/params
├── run_reproduction.py          # One-command reproduction
├── decision_log.md              # 14 non-obvious decisions
├── docs/
│   ├── taxonomy.md              # Hierarchical schema
│   ├── escalation_policy.md     # Written policy (before tuning)
│   └── annotation_guide.md      # Labeling instructions
├── data/
│   ├── processed/               # Extracted threads + embeddings
│   ├── splits/manifest.json     # Thread -> split assignment
│   └── eval/                    # Labeled train/dev/test sets
├── src/
│   ├── data_pipeline/           # Extract, thread, clean, split
│   ├── intent/classifier.py     # Hierarchical classifier
│   ├── retrieval/               # ChromaDB + thread-aware search
│   ├── generation/              # Reply generator + safety checks
│   ├── escalation/router.py     # Policy engine
│   └── agent.py                 # Main orchestrator
├── eval/
│   ├── metrics.py               # Classification + reply + escalation
│   ├── llm_judge.py             # 5-dimension rubric + pairwise
│   ├── safety_audit.py          # Automated safety scanner
│   ├── coverage_risk.py         # Threshold calibration
│   ├── baselines/               # Trivial + simple baselines
│   └── run_eval.py              # Full harness
└── results/                     # Outputs, plots, cached API calls
```

---

## References & Citations

- **Dataset:** Customer Support on Twitter (Kaggle, thoughtvector/customer-support-on-twitter)
- **Embedding model:** sentence-transformers/all-MiniLM-L6-v2
- **Vector store:** ChromaDB
- **LLM:** Configurable (default: OpenAI GPT-4o-mini for generation, GPT-4o for judge)
- AI coding assistants were used freely during development.
