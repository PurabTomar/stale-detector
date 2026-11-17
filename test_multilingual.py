"""
test_multilingual.py - Unit tests for Hindi (Devanagari) and Hinglish extraction,
English search query generation, judge verification, and saved demo examples.

Run with:
    python -m unittest test_multilingual.py -v
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from demo_data import find_cached_pasted_audit, get_multilingual_saved_examples
from extractor import EXTRACTION_SYSTEM_PROMPT, extract_claims
from verifier import VERIFICATION_SYSTEM_PROMPT, verify_claim


class TestMultilingualHindiHinglish(unittest.TestCase):
    def test_extraction_prompt_contains_multilingual_directives(self):
        self.assertIn("HINDI & HINGLISH", EXTRACTION_SYSTEM_PROMPT)
        self.assertIn("VERBATIM in Hindi/Hinglish", EXTRACTION_SYSTEM_PROMPT)
        self.assertIn("search query MUST ALWAYS BE IN ENGLISH", EXTRACTION_SYSTEM_PROMPT)

    def test_verification_prompt_contains_multilingual_directive(self):
        self.assertIn("MULTILINGUAL DIRECTIVE (HINDI AND HINGLISH)", VERIFICATION_SYSTEM_PROMPT)
        self.assertIn("All JSON output keys, \"status\", and \"evidence_stance\" MUST be in English", VERIFICATION_SYSTEM_PROMPT)

    @patch("extractor.call_llm")
    def test_extract_claims_preserves_hindi_text_with_english_query(self, mock_invoke):
        mock_payload = (
            '['
            '  {'
            '    "claim": "आईफोन 15 की शुरुआती कीमत ₹79,900 है", '
            '    "span": "आईफोन 15 की शुरुआती कीमत ₹79,900 है", '
            '    "category": "price", '
            '    "search_query": "iPhone 15 starting price India official Apple", '
            '    "time_sensitivity": 90, '
            '    "volatility_class": "price", '
            '    "reason": "Electronics prices change over release cycles."'
            '  }'
            ']'
        )
        mock_invoke.return_value = (mock_payload, "mock-gemini-model")

        hindi_text = "आईफोन 15 की शुरुआती कीमत ₹79,900 है। यह भारत में उपलब्ध है।"
        claims, model = extract_claims(hindi_text)

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["claim"], "आईफोन 15 की शुरुआती कीमत ₹79,900 है")
        self.assertEqual(claims[0]["span"], "आईफोन 15 की शुरुआती कीमत ₹79,900 है")
        self.assertEqual(claims[0]["category"], "price")
        # Ensure search query is in English
        self.assertEqual(claims[0]["search_query"], "iPhone 15 starting price India official Apple")
        self.assertTrue(all(ord(c) < 128 for c in claims[0]["search_query"].replace("₹", "")))

    @patch("extractor.call_llm")
    def test_extract_claims_preserves_hinglish_text_with_english_query(self, mock_invoke):
        mock_payload = (
            '['
            '  {'
            '    "claim": "iPhone 15 Pro Max ka starting price $1199 tha", '
            '    "span": "iPhone 15 Pro Max ka starting price $1199 tha", '
            '    "category": "price", '
            '    "search_query": "iPhone 15 Pro Max starting price official Apple", '
            '    "time_sensitivity": 85, '
            '    "volatility_class": "price", '
            '    "reason": "Smartphones update annually."'
            '  }'
            ']'
        )
        mock_invoke.return_value = (mock_payload, "mock-gemini-model")

        hinglish_text = "iPhone 15 Pro Max ka starting price $1199 tha. Ye flagship phone hai."
        claims, model = extract_claims(hinglish_text)

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["claim"], "iPhone 15 Pro Max ka starting price $1199 tha")
        self.assertEqual(claims[0]["search_query"], "iPhone 15 Pro Max starting price official Apple")

    def test_saved_demo_cache_contains_real_hindi_and_hinglish_examples(self):
        samples = get_multilingual_saved_examples()
        sample_ids = [s.get("id") for s in samples]

        self.assertIn("hinglish_iphone", sample_ids)
        self.assertIn("hindi_gst", sample_ids)

        hinglish_sample, _ = find_cached_pasted_audit("Apple ne iPhone 15 Pro Max ko $1,199 ke starting price par launch kiya tha")
        self.assertEqual(hinglish_sample["id"], "hinglish_iphone")
        self.assertIn("results", hinglish_sample)
        self.assertEqual(len(hinglish_sample["results"]), 2)
        self.assertEqual(hinglish_sample["results"][0]["status"], "fresh")
        self.assertEqual(hinglish_sample["results"][1]["status"], "outdated")
        self.assertEqual(hinglish_sample["results"][0]["source_origin"], "saved")

        hindi_sample, _ = find_cached_pasted_audit("भारत में मानक जीएसटी दरें 5%, 12%, 18% और 28% हैं")
        self.assertEqual(hindi_sample["id"], "hindi_gst")
        self.assertIn("results", hindi_sample)
        self.assertEqual(len(hindi_sample["results"]), 2)
        self.assertEqual(hindi_sample["results"][0]["status"], "fresh")
        self.assertEqual(hindi_sample["results"][1]["status"], "fresh")
        self.assertEqual(hindi_sample["results"][0]["source_origin"], "saved")


if __name__ == "__main__":
    unittest.main()
