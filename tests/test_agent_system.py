"""
test_agent_system.py -- Comprehensive test suite covering the full SpotifyCares agent system.

Tests:
  1. Text Normalization & Cleaning (clean_text, initials extraction)
  2. Intent Classification (Route, Domain, Risk Flag)
  3. Thread-Aware Retrieval & Leakage Prevention (Strict thread exclusion check)
  4. Generation & Safety Audit (PII, URL, Fabricated refunds, 280-char limit)
  5. Escalation Policy Engine (Hard rules + soft confidence calibrated rules)
  6. End-to-End Agent Execution (process_message pipeline)
  7. Baselines (TrivialBaseline & SimpleBaseline)
  8. Evaluation Metrics, Coverage-Risk & Grounding Audit
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_pipeline.text_cleaner import clean_text, extract_agent_initials
from src.intent.classifier import classify_message
from src.retrieval.retriever import retrieve, retrieval_evidence_strength
from src.generation.reply_generator import generate_reply, _safety_check
from src.escalation.router import decide_escalation
from src.agent import SpotifyCaresAgent
from eval.baselines.trivial_baseline import TrivialBaseline
from eval.baselines.simple_baseline import SimpleBaseline
from eval.metrics import classification_metrics, escalation_metrics
from eval.safety_audit import run_safety_audit, audit_single_reply
from eval.coverage_risk import compute_coverage_risk_curve
from eval.grounding_audit import audit_reply_grounding, run_grounding_audit
from eval.judge_validation import validate_rubric_alignment, validate_pairwise_alignment


class TestDataCleaning(unittest.TestCase):
    def test_clean_text_normalizations(self):
        raw = "@SpotifyCares &amp; @user my music stopped playing &lt;3"
        cleaned = clean_text(raw)
        self.assertIn("&", cleaned)
        self.assertNotIn("&amp;", cleaned)
        self.assertIn("@user", cleaned)
        self.assertNotIn("@SpotifyCares", cleaned)

    def test_extract_initials(self):
        self.assertEqual(extract_agent_initials("We can help /AY"), "AY")
        self.assertEqual(extract_agent_initials("Keep rocking /CP"), "CP")
        self.assertIsNone(extract_agent_initials("No initials here"))


class TestIntentClassification(unittest.TestCase):
    def test_route_classification(self):
        feedback = classify_message("Thanks so much, you guys are the best!")
        self.assertEqual(feedback["route"], "feedback")

        support = classify_message("My Spotify keeps crashing when I open playlists")
        self.assertEqual(support["route"], "support")

    def test_domain_classification(self):
        crash = classify_message("The app keeps crashing on my iPhone")
        self.assertEqual(crash["domain"], "app_device")

        login = classify_message("I cannot log in to my account, forgot password")
        self.assertEqual(login["domain"], "account")

        billing = classify_message("Why was I charged twice for premium this month?")
        self.assertEqual(billing["domain"], "billing")

    def test_risk_flags(self):
        security = classify_message("Someone hacked my account and changed the email!")
        self.assertEqual(security["risk_flag"], "security")

        legal = classify_message("I will sue you for gdpr violations with my lawyer")
        self.assertEqual(legal["risk_flag"], "legal")

        dispute = classify_message("I demand a refund for this unauthorized charge")
        self.assertEqual(dispute["risk_flag"], "payment_dispute")


class TestRetrievalAndLeakagePrevention(unittest.TestCase):
    def test_retrieval_returns_results(self):
        results = retrieve("Spotify keeps crashing on iPhone", k=3)
        self.assertIsInstance(results, list)
        if results:
            self.assertIn("customer_msg", results[0])
            self.assertIn("agent_response", results[0])
            self.assertIn("similarity_score", results[0])

    def test_leakage_prevention_excludes_thread(self):
        """CRITICAL: Ensure queried thread_id is strictly excluded from results."""
        results = retrieve("music skipping", k=5)
        if results:
            target_tid = results[0]["thread_id"]
            # Re-query with that thread explicitly excluded
            filtered_results = retrieve("music skipping", exclude_thread_id=target_tid, k=5)
            filtered_tids = [r["thread_id"] for r in filtered_results]
            self.assertNotIn(
                target_tid,
                filtered_tids,
                f"Data leakage detected! Excluded thread {target_tid} was returned in retrieval.",
            )


class TestSafetyAuditAndGeneration(unittest.TestCase):
    def test_safety_check_detects_violations(self):
        # 1. PII violation
        pii_reply = "Send us your password and credit card number so we can fix it /AI"
        pii_scan = audit_single_reply(pii_reply)
        self.assertFalse(pii_scan["passed"])
        self.assertGreater(pii_scan["categories"]["pii_request"], 0)

        # 2. Fabricated refund violation
        refund_reply = "We will refund $9.99 back to your bank account /AI"
        refund_scan = audit_single_reply(refund_reply)
        self.assertFalse(refund_scan["passed"])
        self.assertGreater(refund_scan["categories"]["fabricated_refund"], 0)

        # 3. Safe reply passes
        safe_reply = "Hi there! Could you try restarting your device to see if that helps? /AI"
        safe_scan = audit_single_reply(safe_reply)
        self.assertTrue(safe_scan["passed"])

    def test_generate_reply_structure(self):
        cls_info = {"route": "support", "domain": "playback", "risk_flag": "none"}
        retrieved = [{"customer_msg": "sound issue", "agent_response": "restart app", "similarity_score": 0.8}]
        gen = generate_reply("My sound is distorted", cls_info, retrieved, evidence_strength="moderate")
        self.assertIn("reply", gen)
        self.assertLessEqual(len(gen["reply"]), config.TWITTER_CHAR_LIMIT)
        self.assertIn("/AI", gen["reply"])


class TestEscalationRouter(unittest.TestCase):
    def test_hard_rules_escalation(self):
        # Security -> always escalate
        res_sec = decide_escalation({"route": "support", "domain": "account", "risk_flag": "security", "confidence": 0.95})
        self.assertEqual(res_sec["decision"], "escalate")

        # Payment dispute -> always escalate
        res_pay = decide_escalation({"route": "support", "domain": "billing", "risk_flag": "payment_dispute", "confidence": 0.90})
        self.assertEqual(res_pay["decision"], "escalate")

        # Repeated contact (thread >= 5) -> escalate
        res_rep = decide_escalation({"route": "support", "domain": "playback", "risk_flag": "none", "confidence": 0.90}, thread_length=6)
        self.assertEqual(res_rep["decision"], "escalate")

    def test_auto_handle_rules(self):
        # Routine feedback -> auto_handle
        res_fb = decide_escalation({"route": "feedback", "domain": None, "risk_flag": "none", "confidence": 0.95})
        self.assertEqual(res_fb["decision"], "auto_handle")

        # High confidence support with strong evidence -> auto_handle
        res_sup = decide_escalation(
            {"route": "support", "domain": "playback", "risk_flag": "none", "confidence": 0.85},
            evidence_strength="strong",
            thread_length=1,
        )
        self.assertEqual(res_sup["decision"], "auto_handle")


class TestEndToEndAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = SpotifyCaresAgent(use_cache=False)

    def test_process_message_pipeline(self):
        msg = "My playlist stopped downloading for offline listening on Android"
        result = self.agent.process_message(msg)

        # Check top-level contract keys
        for key in ["input", "classification", "retrieval", "generation", "escalation"]:
            self.assertIn(key, result)

        cls_res = result["classification"]
        self.assertEqual(cls_res["route"], "support")

        esc_res = result["escalation"]
        self.assertIn(esc_res["decision"], ["auto_handle", "escalate", "clarify"])
        self.assertTrue(len(esc_res["reason"]) > 0)

        gen_res = result["generation"]
        self.assertLessEqual(len(gen_res["reply"]), 280)


class TestBaselines(unittest.TestCase):
    def test_trivial_baseline(self):
        trivial = TrivialBaseline()
        res = trivial.process_message("How do I cancel my subscription?")
        self.assertEqual(res["classification"]["route"], "support")
        self.assertEqual(res["escalation"]["decision"], "auto_handle")

    def test_simple_baseline(self):
        simple = SimpleBaseline(config.TRAINING_LABELS_JSONL)
        res = simple.process_message("App crashing every morning on iPhone")
        self.assertIn(res["classification"]["route"], ["support", "feedback", "abuse_spam"])
        self.assertIn("escalation", res)


class TestMetricsAndAudits(unittest.TestCase):
    def test_classification_metrics(self):
        y_true = ["support", "support", "feedback", "abuse_spam"]
        y_pred = ["support", "support", "feedback", "support"]
        m = classification_metrics(y_true, y_pred, "route")
        self.assertEqual(m["accuracy"], 0.75)
        self.assertIn("macro_f1", m)

    def test_coverage_risk_curve(self):
        confs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        correct = [True, True, True, False, False, False]
        curve_res = compute_coverage_risk_curve(confs, correct)
        self.assertIn("recommended_threshold", curve_res)
        self.assertIn("curve", curve_res)

    def test_grounding_audit(self):
        reply = "Try clearing your cache and restarting your phone /AI"
        retrieved = [{"customer_msg": "cache issue", "agent_response": "clearing your cache helps", "similarity_score": 0.8}]
        audit = audit_reply_grounding(reply, retrieved)
        self.assertIn("grounding_score", audit)
        self.assertIn("status", audit)


if __name__ == "__main__":
    unittest.main(verbosity=2)
