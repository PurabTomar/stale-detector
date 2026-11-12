"""
test_extractor.py - Unit tests for claim extraction & JSON parsing in extractor.py and llm.py.

Run with:
    python -m unittest test_extractor.py -v
"""

import unittest
from unittest.mock import patch

import extractor
import llm


class TestExtractionJSONParsing(unittest.TestCase):
    """Tests for JSON parsing, fenced/prose cleanup, schema normalization, and error paths."""

    def setUp(self):
        llm.pop_last_gateway_metrics()

    def test_parse_json_response_variants(self):
        # 1. Clean JSON array
        raw_array = '[{"claim": "Python 3.14 is latest.", "category": "version", "search_query": "latest python"}]'
        self.assertEqual(len(llm.parse_json_response(raw_array)), 1)

        # 2. Markdown code-fenced JSON (lowercase/uppercase tag or bare fence)
        fenced = '```json\n[\n  {"claim": "iPhone starts at $999.", "category": "price", "search_query": "iPhone price"}\n]\n```'
        parsed_fenced = llm.parse_json_response(fenced)
        self.assertEqual(parsed_fenced[0]["category"], "price")

        # 3. Leading & trailing explanatory prose around JSON array
        with_prose = (
            "Sure! Here are the time-sensitive claims extracted:\n"
            '[{"claim": "CEO is Jane Doe.", "category": "company", "search_query": "Company CEO"}]\n'
            "Let me know if you need anything else."
        )
        parsed_prose = llm.parse_json_response(with_prose)
        self.assertEqual(len(parsed_prose), 1)
        self.assertEqual(parsed_prose[0]["claim"], "CEO is Jane Doe.")

        # 4. Embedded JSON object
        wrapped = 'Verdict:\n{"status": "fresh", "score": 92}'
        self.assertEqual(llm.parse_json_response(wrapped), {"status": "fresh", "score": 92})

    def test_parse_json_response_invalid_inputs_raise_value_error(self):
        for bad_input in [
            "",
            "   \n  ",
            "Sorry, I could not find any claims to extract.",
            '[{"claim": "Incomplete json"',
            "{broken: json without quotes}",
        ]:
            with self.subTest(bad_input=bad_input):
                with self.assertRaises(ValueError):
                    llm.parse_json_response(bad_input)

    def test_normalize_claims_handles_wrappers_categories_and_defaults(self):
        # 1. Wrapped object like {"claims": [...]} unwraps cleanly
        wrapped_payload = {
            "claims": [
                {"claim": "  GST rate is 18%. ", "category": "REGULATION", "search_query": "India GST rate"},
                {"claim": "Headcount is 500.", "category": "bogus_category", "search_query": ""},
                {"claim": "   ", "category": "price", "search_query": "ignored blank claim"},
                "not-a-dict-entry",
            ]
        }
        normalized = extractor._normalize_claims(wrapped_payload)
        self.assertEqual(len(normalized), 2)
        self.assertEqual(
            normalized[0],
            {"claim": "GST rate is 18%.", "category": "regulation", "search_query": "India GST rate"},
        )
        self.assertEqual(
            normalized[1],
            {"claim": "Headcount is 500.", "category": "other", "search_query": "Headcount is 500."},
        )

        # 2. Root scalar or dict without any list rejects cleanly with ValueError
        for invalid_root in [42, "just a string", {"message": "no list here"}]:
            with self.subTest(invalid_root=invalid_root):
                with self.assertRaises(ValueError):
                    extractor._normalize_claims(invalid_root)

    def test_extract_claims_end_to_end_and_error_handling(self):
        # Blank input skips LLM call cleanly
        claims_empty, model_empty = extractor.extract_claims("   ")
        self.assertEqual(claims_empty, [])
        self.assertEqual(model_empty, "")
        self.assertIsNone(extractor.get_last_gateway_metrics())

        # Valid fenced response attaches normalized claims + operation telemetry
        fake_response = """Here is the extracted list:
        ```json
        [
          {"claim": "Python 3.11 is the newest version.", "category": "VERSION", "search_query": "latest stable Python version"},
          {"claim": "OpenAI GPT-4 costs $30 per 1M output tokens.", "category": "Price", "search_query": "OpenAI GPT-4 pricing"}
        ]
        ```"""

        def fake_call_llm(system_prompt, user_prompt, temperature=0.1):
            llm._record_gateway_metrics(
                {
                    "primary_model": "gemini-2.5-flash",
                    "model_used": "gemini-2.5-flash",
                    "fallback_used": False,
                    "latency_ms": 95.0,
                    "prompt_tokens": 70,
                    "completion_tokens": 30,
                    "total_tokens": 100,
                }
            )
            return fake_response, "gemini-2.5-flash"

        with patch("extractor.call_llm", side_effect=fake_call_llm):
            claims, model_used = extractor.extract_claims("Python 3.11 and OpenAI GPT-4...")

        self.assertEqual(model_used, "gemini-2.5-flash")
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0]["category"], "version")
        self.assertEqual(claims[1]["category"], "price")
        metrics = extractor.get_last_gateway_metrics()
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["operation"], "Extract claims")
        self.assertEqual(metrics["total_tokens"], 100)

        # Malformed LLM JSON raises clear ValueError
        with patch("extractor.call_llm", return_value=("Not valid JSON at all", "gemini-2.5-flash")):
            with self.assertRaises(ValueError) as ctx:
                extractor.extract_claims("Some claim text.")
        self.assertIn("JSON", str(ctx.exception))

    def test_extract_claims_routes_with_extract_task(self):
        captured_kwargs: dict = {}

        def task_aware_call_llm(system_prompt, user_prompt, temperature=0.1, task=None):
            captured_kwargs["task"] = task
            llm._record_gateway_metrics(
                {
                    "primary_model": "gemini/gemini-2.5-flash-lite",
                    "model_used": "gemini/gemini-2.5-flash-lite",
                    "fallback_used": False,
                    "latency_ms": 48.0,
                    "prompt_tokens": 30,
                    "completion_tokens": 12,
                    "total_tokens": 42,
                    "task": task,
                }
            )
            return (
                '[{"claim": "Python 3.14 is latest.", "category": "version", "search_query": "Python latest"}]',
                "gemini/gemini-2.5-flash-lite",
            )

        with patch("extractor.call_llm", side_effect=task_aware_call_llm):
            claims, model_used = extractor.extract_claims("Python 3.14 is latest.")

        self.assertEqual(captured_kwargs.get("task"), "extract")
        self.assertEqual(model_used, "gemini/gemini-2.5-flash-lite")
        self.assertEqual(len(claims), 1)
        metrics = extractor.get_last_gateway_metrics()
        self.assertEqual(metrics["task"], "extract")
        self.assertEqual(metrics["operation"], "Extract claims")

    def test_span_extraction_normalization_and_locate_span_in_text(self):
        source = (
            "Apple currently charges $1,199 for the flagship iPhone Pro Max. "
            "Meanwhile, Python 3.13 is available, and Python 3.13 supports free-threaded builds."
        )

        # 1. Exact literal match returns exact slice
        loc1 = extractor.locate_span_in_text(
            source,
            span_text="$1,199 for the flagship iPhone Pro Max",
            claim_text="Apple charges $1,199 for iPhone Pro Max.",
        )
        self.assertIsNotNone(loc1)
        self.assertEqual(source[loc1[0]:loc1[1]], "$1,199 for the flagship iPhone Pro Max")

        # 2. Whitespace-collapsed & case-insensitive span variant resolves exact slice in source
        loc_ws = extractor.locate_span_in_text(
            "Line one has   Python\n3.13   in production.",
            span_text="python 3.13 in production",
        )
        self.assertIsNotNone(loc_ws)
        self.assertEqual(
            "Line one has   Python\n3.13   in production."[loc_ws[0]:loc_ws[1]],
            "Python\n3.13   in production",
        )

        # 3. Duplicate occurrence avoids occupied_ranges and finds second match
        dup_text = "Tier A costs $20 per month, and Tier B costs $20 per month."
        first = extractor.locate_span_in_text(dup_text, span_text="costs $20 per month")
        self.assertIsNotNone(first)
        second = extractor.locate_span_in_text(
            dup_text,
            span_text="costs $20 per month",
            occupied_ranges=[(first[0], first[1])],
        )
        self.assertIsNotNone(second)
        self.assertGreater(second[0], first[1])
        self.assertEqual(dup_text[second[0]:second[1]], "costs $20 per month")

        # 4. _normalize_claims with source_text recovers verbatim spans and respects occupied ranges
        raw = [
            {
                "claim": "Tier A costs $20/mo.",
                "span": "Tier A costs $20 per month",
                "category": "price",
                "search_query": "Tier A price",
            },
            {
                "claim": "Tier B costs $20/mo.",
                "span": "costs $20 per month",
                "category": "price",
                "search_query": "Tier B price",
            },
        ]
        norm = extractor._normalize_claims(raw, source_text=dup_text)
        self.assertEqual(norm[0]["span"], "Tier A costs $20 per month")
        self.assertEqual(norm[1]["span"], "costs $20 per month")

        # 5. extract_claims attaches exact verbatim span from source input
        fake_json = (
            '[{"claim": "iPhone Pro Max costs $1,199", '
            '"span": "charges $1,199 for the flagship iPhone Pro Max", '
            '"category": "price", "search_query": "iPhone Pro Max price"}]'
        )
        with patch("extractor.call_llm", return_value=(fake_json, "gemini/gemini-2.5-flash-lite")):
            extracted, _ = extractor.extract_claims(source)
        self.assertEqual(extracted[0]["span"], "charges $1,199 for the flagship iPhone Pro Max")

    def test_normalize_claims_deduplicates_duplicate_claims_while_preserving_distinct(self):
        raw = [
            {
                "claim": "Python 3.11 is the latest release.",
                "span": "Python 3.11 is the latest release",
                "category": "version",
                "search_query": "latest Python version",
            },
            {
                "claim": "  python   3.11 is the latest release ",
                "span": "Python 3.11 is the latest release",
                "category": "version",
                "search_query": "Python newest",
            },
            {
                "claim": "iOS 17 is the latest iOS.",
                "span": "iOS 17 is the latest iOS",
                "category": "version",
                "search_query": "latest iOS",
            },
        ]
        norm = extractor._normalize_claims(
            raw,
            source_text="Python 3.11 is the latest release. iOS 17 is the latest iOS.",
        )
        self.assertEqual(len(norm), 2)
        self.assertEqual(norm[0]["claim"], "Python 3.11 is the latest release.")
        self.assertEqual(norm[1]["claim"], "iOS 17 is the latest iOS.")


if __name__ == "__main__":
    unittest.main()
