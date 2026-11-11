"""
test_demo.py - Unit tests for offline Demo mode cache (`demo_data.py` & `demo_cache.json`).

Ensures all pre-verified sample audits (Audit pasted answer, Ask & Audit, and Audit a URL)
load offline with 0 network/API calls, match verbatim spans inside source text, satisfy
score-band invariants, and return defensive deep copies.
"""

import unittest

import demo_data
import verifier


class TestDemoCache(unittest.TestCase):
    def test_load_demo_cache_structure_and_pasted_examples(self):
        cache = demo_data.load_demo_cache()
        self.assertIn("pasted_examples", cache)
        self.assertIn("ask_sample", cache)
        self.assertIn("url_sample", cache)

        pasted = demo_data.get_pasted_demo_examples()
        self.assertEqual(len(pasted), 4)

        for idx, sample in enumerate(pasted):
            ans = sample.get("answer_text", "")
            self.assertTrue(ans.strip())
            self.assertTrue(str(sample.get("corrected_answer") or "").strip())
            self.assertNotEqual(ans.strip(), sample["corrected_answer"].strip())
            results = sample.get("results") or []
            self.assertTrue(len(results) >= 2, f"Sample {idx} should have >=2 claims")
            for claim_row in results:
                span = claim_row.get("span") or ""
                self.assertIn(span, ans, f"Verbatim span missing in pasted example {idx}: {span!r}")
                lo, hi = verifier.STATUS_SCORE_BANDS[claim_row["status"]]
                if lo is None or hi is None:
                    self.assertIsNone(claim_row["score"])
                else:
                    self.assertTrue(lo <= claim_row["score"] <= hi)
                self.assertEqual(claim_row.get("tavily_credits_used"), 0.0)

    def test_find_cached_pasted_audit_matching_and_fallback(self):
        # Exact match on every example text
        examples = demo_data.get_pasted_demo_examples()
        for ex in examples:
            found, exact = demo_data.find_cached_pasted_audit(ex["answer_text"])
            self.assertTrue(exact)
            self.assertEqual(found["id"], ex["id"])

        # Keyword-assisted match on shortened snippet
        found_gst, exact_gst = demo_data.find_cached_pasted_audit("India GST tax slabs include 28%.")
        self.assertTrue(exact_gst)
        self.assertEqual(found_gst["id"], "gst")

        # Unknown custom prompt gracefully falls back to sample[0] with exact_match=False
        fallback, exact_custom = demo_data.find_cached_pasted_audit(
            "An entirely unrelated hypothetical widget factory quote."
        )
        self.assertFalse(exact_custom)
        self.assertEqual(fallback["id"], examples[0]["id"])

    def test_ask_and_url_demo_samples_spans_and_defensive_copies(self):
        ask_sample = demo_data.get_cached_ask_audit("Custom question about RBI repo rate")
        self.assertEqual(ask_sample["question"], "Custom question about RBI repo rate")
        self.assertTrue(ask_sample["answer_text"].strip())
        self.assertTrue(ask_sample["corrected_answer"].strip())
        for claim_row in ask_sample["results"]:
            self.assertIn(claim_row["span"], ask_sample["answer_text"])
            lo, hi = verifier.STATUS_SCORE_BANDS[claim_row["status"]]
            self.assertTrue(lo <= claim_row["score"] <= hi)

        # Mutating returned results does not pollute future calls
        ask_sample["results"][0]["status"] = "mutated"
        ask_fresh = demo_data.get_cached_ask_audit()
        self.assertEqual(ask_fresh["question"], ask_fresh["default_question"])
        self.assertNotEqual(ask_fresh["results"][0]["status"], "mutated")

        url_sample = demo_data.get_cached_url_audit("https://blog.example.com/custom-article")
        self.assertEqual(url_sample["url"], "https://blog.example.com/custom-article")
        self.assertTrue(url_sample["content"].strip())
        self.assertTrue(url_sample["corrected_answer"].strip())
        for claim_row in url_sample["results"]:
            self.assertIn(claim_row["span"], url_sample["content"])
            lo, hi = verifier.STATUS_SCORE_BANDS[claim_row["status"]]
            self.assertTrue(lo <= claim_row["score"] <= hi)

        url_default = demo_data.get_cached_url_audit()
        self.assertEqual(url_default["url"], url_default["default_url"])


if __name__ == "__main__":
    unittest.main()
