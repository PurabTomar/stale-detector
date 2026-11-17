"""
test_batch_audit.py - Unit tests for batch_audit.py parsing, safety limits,
summarization, and CSV/JSON export.

Run with:
    python -m unittest test_batch_audit.py -v
"""

from __future__ import annotations

import json
import unittest

from batch_audit import (
    MAX_BATCH_FILE_SIZE_BYTES,
    MAX_BATCH_ITEMS,
    export_batch_summary_csv,
    export_batch_summary_json,
    parse_batch_input,
    summarize_batch_item,
)


class TestBatchAudit(unittest.TestCase):
    def test_parse_batch_pasted_text_blank_lines(self):
        text = "Answer 1 about iPhone.\n\nAnswer 2 about GPT-4.\n\n\nAnswer 3 about GST."
        items, err = parse_batch_input(raw_text=text)
        self.assertIsNone(err)
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0], "Answer 1 about iPhone.")
        self.assertEqual(items[1], "Answer 2 about GPT-4.")
        self.assertEqual(items[2], "Answer 3 about GST.")

    def test_parse_batch_csv_with_header(self):
        csv_bytes = b"answer,source\n\"The iPhone 15 costs $799.\",apple\n\"GPT-4 is released.\",openai"
        items, err = parse_batch_input(file_bytes=csv_bytes, filename="test.csv")
        self.assertIsNone(err)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0], "The iPhone 15 costs $799.")
        self.assertEqual(items[1], "GPT-4 is released.")

    def test_parse_batch_csv_without_header(self):
        csv_bytes = b"\"Price is $100.\",extra\n\"Price is $200.\",extra"
        items, err = parse_batch_input(file_bytes=csv_bytes, filename="data.csv")
        self.assertIsNone(err)
        self.assertEqual(len(items), 2)
        self.assertIn("Price is $100.", items[0])

    def test_parse_batch_file_size_limit_200kb(self):
        oversized = b"A" * (MAX_BATCH_FILE_SIZE_BYTES + 500)
        items, err = parse_batch_input(file_bytes=oversized, filename="large.txt")
        self.assertIsNotNone(err)
        self.assertIn("200 KB", err)
        self.assertEqual(items, [])

    def test_parse_batch_max_20_items_cap(self):
        chunks = [f"Item {i} test answer fact statement." for i in range(35)]
        text = "\n\n".join(chunks)
        items, err = parse_batch_input(raw_text=text)
        self.assertIsNone(err)
        self.assertEqual(len(items), MAX_BATCH_ITEMS)
        self.assertEqual(items[0], "Item 0 test answer fact statement.")
        self.assertEqual(items[19], "Item 19 test answer fact statement.")

    def test_summarize_batch_item(self):
        results = [
            {"claim": "iPhone 15 is current model", "status": "outdated", "score": 20},
            {"claim": "Apple released iOS 17", "status": "fresh", "score": 90},
        ]
        summary = summarize_batch_item(
            index=1,
            answer_text="Apple sells the iPhone 15 as its latest model.",
            results=results,
            overall_score=55,
            source_origin="live",
        )
        self.assertEqual(summary["index"], 1)
        self.assertEqual(summary["overall_score"], 55)
        self.assertEqual(summary["fresh_count"], 1)
        self.assertEqual(summary["outdated_count"], 1)
        self.assertEqual(summary["total_claims"], 2)
        self.assertIn("iPhone 15", summary["worst_claim"])
        self.assertEqual(summary["worst_status"], "outdated")
        self.assertEqual(summary["source_label"], "Live")

    def test_export_csv_and_json(self):
        row = {
            "index": 1,
            "snippet": "Test snippet...",
            "overall_score": 85,
            "fresh_count": 2,
            "outdated_count": 0,
            "unverifiable_count": 0,
            "total_claims": 2,
            "worst_claim": "None",
            "worst_status": "fresh",
            "source_label": "Cached",
        }
        csv_str = export_batch_summary_csv([row])
        self.assertIn("Item #", csv_str)
        self.assertIn("Test snippet...", csv_str)
        self.assertIn("Cached", csv_str)

        json_str = export_batch_summary_json([row])
        data = json.loads(json_str)
        self.assertEqual(data["total_items"], 1)
        self.assertEqual(data["batch_summary"][0]["overall_score"], 85)


if __name__ == "__main__":
    unittest.main()
