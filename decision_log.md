# Decision Log

Non-obvious decisions made during the design and implementation of this project.

---

## 1. Brand Selection: SpotifyCares (with DM-redirect caveat)

**Chose** SpotifyCares over AmazonHelp (169K tweets, too generic), AppleSupport (106K, template-heavy), and Delta (42K, temporal bias).

**Why:** 43K outbound tweets with diagnostic questioning pattern (45%), KB link referrals (50.5%), and a distinctive brand voice.

**Caveat:** 30.8% of SpotifyCares responses redirect to DM -- the actual resolution is private. We frame retrieval as "historical public support patterns," not verified resolutions.

## 2. Hierarchical Schema (Route/Domain/Risk) over flat intent list

**Rejected** a flat 10-intent taxonomy because intents overlap (playback + app crash + device compatibility co-occur). A hierarchical schema separates *what the issue is* (domain) from *how to handle it* (risk flag), making escalation decisions cleaner.

## 3. Thread-level chronological split (not random, not tweet-level)

**Why thread-level:** Splitting individual tweets across sets would leak thread context into the test set.

**Why chronological:** Simulates production deployment where the model sees only past data. Random splits would intermix temporal patterns.

**Result:** Train (17,256 threads, before Nov 10) / Dev (4,984, Nov 10-20) / Test (7,245, Nov 21+). Zero leakage verified.

## 4. Written escalation policy before tuning

The policy was defined as a document (`docs/escalation_policy.md`) before any model work, so that evaluation labels are grounded in explicit rules rather than post-hoc model-influenced decisions.

## 5. Three separate label sets (train/dev/frozen test)

TF-IDF baseline trains on training labels (300-500 examples). Thresholds and prompts are tuned on dev labels (100-150). Frozen test set (200) is touched exactly once for final reported results. Prevents label leakage from tuning.

## 6. "Evidence-supportedness" instead of "Accuracy"

We have no authoritative Spotify product knowledge source. We cannot judge if a recommendation is objectively *correct*, only whether it is *supported by retrieved historical examples*. Renamed the judge dimension accordingly.

## 7. Safety audit as first-class evaluation dimension

Not an afterthought. The agent cannot access accounts, issue refunds, or take actions. Any reply claiming such capabilities is a safety violation. Automated scanner checks every generated reply.

## 8. Coverage-risk curve instead of single confidence threshold

Raw LLM confidence is not calibrated. Instead of trusting a fixed threshold, we compute a coverage-risk curve on the dev set: "at threshold T, the agent auto-handles X% of cases with Y% error." This lets evaluators choose their risk appetite.

## 9. Different models for generation vs. judge

Using the same model family for generation and evaluation introduces circular bias (the judge prefers outputs that "sound right" to that model). Default: GPT-4o-mini for generation, GPT-4o for judge.

## 10. Configurable LLM provider (not committed to named models)

Model availability and pricing change. All LLM calls go through `config.py` with env var overrides. The reproduction path uses cached outputs so evaluators don't need API keys.

## 11. Branch preservation (not "keep first response")

3,475 tweets in the dataset have multiple responses. We preserve all branches rather than arbitrarily keeping only the first. For retrieval, all response variants are indexed. This decision is documented explicitly.

## 12. DM-redirect framing (public patterns, not verified resolutions)

The retrieval corpus is framed as "historical public support patterns" -- acknowledging that 30.8% of interactions redirect to private DM where the actual resolution happens. The agent learns triage and initial response patterns, not complete technical support.

## 13. Pairwise comparison as primary reply metric (not BLEU/ROUGE)

Many valid replies exist for one customer tweet. BLEU/ROUGE reward lexical overlap with a single reference, which is misleading. The primary metric is an LLM-judge rubric + pairwise comparison (retrieval-only vs. full RAG). BLEU/ROUGE are reported as secondary diagnostics only.

## 14. What we chose NOT to build (and why)

- **Fine-tuned models:** Hard to reproduce, training time exceeds scope.
- **Multi-brand support:** Depth over breadth.
- **Production deployment:** Not asked for; CLI pipeline suffices.
- **Official Spotify docs integration:** Outside dataset scope (noted as "next week" improvement).
- **Sentiment model training:** Off-the-shelf VADER suffices for escalation signal.
