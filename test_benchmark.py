"""
test_benchmark.py - Unit tests for benchmark/dataset.json validation,
benchmark metric computation, cached results persistence, mocked parallel
benchmark execution, and README table formatting in benchmark.py.

Run with:
    python -m unittest test_benchmark.py -v
"""

from collections import Counter
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import benchmark
import llm


class TestBenchmarkDataset(unittest.TestCase):
    """Validate structure and balance of benchmark/dataset.json."""

    def test_dataset_has_24_valid_labelled_claims(self):
        entries = benchmark.load_dataset()
        self.assertEqual(len(entries), 24)

        status_counts = Counter(e["expected_status"] for e in entries)
        self.assertEqual(status_counts["outdated"], 11)
        self.assertEqual(status_counts["fresh"], 11)
        self.assertEqual(status_counts["uncertain"], 1)
        self.assertEqual(status_counts["unverifiable"], 1)

        cat_counts = Counter(e["category"] for e in entries)
        for cat in ("price", "version", "regulation", "company", "statistic"):
            self.assertGreaterEqual(cat_counts[cat], 4, f"Expected >=4 claims in category {cat!r}")

        self.assertTrue(
            any("iPhone 18 Pro is available" in e["claim"] for e in entries),
            "Dataset should include the 'iPhone 18 Pro is available' regression claim.",
        )

        for idx, entry in enumerate(entries, start=1):
            self.assertEqual(entry["index"], idx)
            self.assertTrue(entry["claim"])
            self.assertTrue(entry["note"])
            self.assertIn(entry["expected_status"], {"fresh", "outdated", "uncertain", "unverifiable"})

    def test_load_dataset_validation_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                benchmark.load_dataset(tmp_dir / "missing.json")

            bad_empty = tmp_dir / "empty.json"
            bad_empty.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                benchmark.load_dataset(bad_empty)

            bad_status = tmp_dir / "bad_status.json"
            bad_status.write_text(
                json.dumps(
                    [
                        {
                            "claim": "Test claim",
                            "category": "price",
                            "expected_status": "bogus_status",
                            "note": "Invalid label for dataset",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                benchmark.load_dataset(bad_status)

            # All 4 supported benchmark statuses load cleanly
            all_four = tmp_dir / "all_four.json"
            all_four.write_text(
                json.dumps(
                    [
                        {"claim": "C1", "category": "price", "expected_status": "fresh", "note": "n1"},
                        {"claim": "C2", "category": "price", "expected_status": "outdated", "note": "n2"},
                        {"claim": "C3", "category": "statistic", "expected_status": "uncertain", "note": "n3"},
                        {"claim": "C4", "category": "version", "expected_status": "unverifiable", "note": "n4"},
                    ]
                ),
                encoding="utf-8",
            )
            loaded_four = benchmark.load_dataset(all_four)
            self.assertEqual([e["expected_status"] for e in loaded_four], ["fresh", "outdated", "uncertain", "unverifiable"])

            bad_claim = tmp_dir / "bad_claim.json"
            bad_claim.write_text(
                json.dumps([{"claim": "   ", "category": "price", "expected_status": "fresh"}]),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                benchmark.load_dataset(bad_claim)


class TestBenchmarkMetricsAndCache(unittest.TestCase):
    """Tests for accuracy, stale precision/recall/F1, confusion matrix, and JSON cache."""

    def test_compute_metrics_empty_and_mixed_cases(self):
        empty = benchmark.compute_benchmark_metrics([], total_duration_ms=0.0)
        self.assertEqual(empty["total_claims"], 0)
        self.assertEqual(empty["accuracy"], 0.0)
        self.assertEqual(empty["precision_stale"], 0.0)
        self.assertEqual(empty["recall_stale"], 0.0)
        self.assertEqual(empty["confusion_matrix"]["tp"], 0)

        rows = [
            # TP (outdated -> outdated)
            {
                "category": "price",
                "expected_status": "outdated",
                "predicted_status": "outdated",
                "latency_ms": 100.0,
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "total_tokens": 70,
                "tavily_credits_used": 2.0,
            },
            # TP (outdated -> outdated)
            {
                "category": "version",
                "expected_status": "outdated",
                "predicted_status": "outdated",
                "latency_ms": 200.0,
                "prompt_tokens": 60,
                "completion_tokens": 30,
                "total_tokens": 90,
                "tavily_credits_used": 1.0,
            },
            # FN (outdated -> fresh)
            {
                "category": "regulation",
                "expected_status": "outdated",
                "predicted_status": "fresh",
                "latency_ms": 300.0,
                "prompt_tokens": 40,
                "completion_tokens": 10,
                "total_tokens": 50,
                "tavily_credits_used": 1.2,
            },
            # FN into other (outdated -> uncertain)
            {
                "category": "company",
                "expected_status": "outdated",
                "predicted_status": "uncertain",
                "latency_ms": 400.0,
                "prompt_tokens": 30,
                "completion_tokens": 20,
                "total_tokens": 50,
                "tavily_credits_used": 0.0,
            },
            # FP (fresh -> outdated)
            {
                "category": "price",
                "expected_status": "fresh",
                "predicted_status": "outdated",
                "latency_ms": 500.0,
                "prompt_tokens": 20,
                "completion_tokens": 20,
                "total_tokens": 40,
                "tavily_credits_used": 2.2,
            },
            # TN (fresh -> fresh)
            {
                "category": "statistic",
                "expected_status": "fresh",
                "predicted_status": "fresh",
                "latency_ms": 600.0,
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "tavily_credits_used": 1.0,
            },
        ]
        m = benchmark.compute_benchmark_metrics(rows, total_duration_ms=950.0)
        self.assertEqual(m["total_claims"], 6)
        self.assertEqual(m["correct"], 3)
        self.assertEqual(m["incorrect"], 3)
        self.assertAlmostEqual(m["accuracy"], 0.5, places=4)
        # TP=2, FP=1, FN=2, TN=1
        self.assertAlmostEqual(m["precision_stale"], round(2 / 3, 4), places=4)
        self.assertAlmostEqual(m["recall_stale"], 0.5, places=4)
        expected_f1 = round((2 * (round(2 / 3, 4)) * 0.5) / (round(2 / 3, 4) + 0.5), 4)
        self.assertAlmostEqual(m["f1_stale"], expected_f1, places=4)
        self.assertEqual(m["avg_latency_ms"], 350.0)
        self.assertEqual(m["total_duration_ms"], 950.0)
        self.assertEqual(m["total_prompt_tokens"], 300)
        self.assertEqual(m["total_completion_tokens"], 150)
        self.assertEqual(m["total_tokens"], 450)
        self.assertAlmostEqual(m["total_tavily_credits"], 7.4, places=2)

        cm = m["confusion_matrix"]
        self.assertEqual((cm["tp"], cm["fp"], cm["fn"], cm["tn"]), (2, 1, 2, 1))
        self.assertEqual(cm["matrix"]["outdated"], {"outdated": 2, "fresh": 1, "other": 1})
        self.assertEqual(cm["matrix"]["fresh"], {"outdated": 1, "fresh": 1, "other": 0})
        self.assertEqual(m["by_category"]["price"]["correct"], 1)
        self.assertEqual(m["by_category"]["price"]["total"], 2)

    def test_cache_load_save_round_trip_and_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_file = Path(tmp) / "results.json"
            self.assertIsNone(benchmark.load_cached_results(cache_file))

            cache_file.write_text("{corrupt json", encoding="utf-8")
            self.assertIsNone(benchmark.load_cached_results(cache_file))

            cache_file.write_text(json.dumps({"not_results": []}), encoding="utf-8")
            self.assertIsNone(benchmark.load_cached_results(cache_file))

            sample_payload = {
                "app": "Stale-Answer Detector",
                "generated_at": "2026-11-11T19:00:00+05:30",
                "routing_mode": "bifrost",
                "models_used": ["gemini/gemini-2.5-flash-lite", "openai/gpt-4o-mini"],
                "metrics": {"total_claims": 1, "correct": 1, "accuracy": 1.0},
                "results": [{"index": 1, "claim": "₹4 lakh threshold", "correct": True}],
            }
            saved_path = benchmark.save_cached_results(sample_payload, results_path=cache_file)
            self.assertEqual(saved_path, cache_file)
            loaded = benchmark.load_cached_results(cache_file)
            self.assertEqual(loaded, sample_payload)


class TestBenchmarkRunnerAndMarkdown(unittest.TestCase):
    """Tests for mocked parallel run_benchmark, outage protection, and README formatting."""

    def test_run_benchmark_parallel_order_cache_and_outage_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            ds_path = tmp_dir / "dataset.json"
            res_path = tmp_dir / "results.json"
            readme_path = tmp_dir / "README.md"

            dataset_entries = [
                {
                    "claim": "Old rate is 6.50%.",
                    "category": "statistic",
                    "expected_status": "outdated",
                    "note": "Now 5.50%",
                },
                {
                    "claim": "ChatGPT Plus costs $20/month.",
                    "category": "price",
                    "expected_status": "fresh",
                    "note": "Still $20/month",
                },
                {
                    "claim": "React 18 is latest.",
                    "category": "version",
                    "expected_status": "outdated",
                    "note": "React 19 is current",
                },
                {
                    "claim": "SEC T+1 settlement applies.",
                    "category": "regulation",
                    "expected_status": "fresh",
                    "note": "T+1 effective May 2024",
                },
            ]
            ds_path.write_text(json.dumps(dataset_entries), encoding="utf-8")
            readme_path.write_text(
                "# Header\n\n"
                f"{benchmark.BENCHMARK_README_START}\nold placeholder\n{benchmark.BENCHMARK_README_END}\n\n"
                "## Footer\n",
                encoding="utf-8",
            )

            seen_use_cache: list[bool] = []
            seen_dual_judge: list[bool] = []

            def fake_extract(text: str):
                llm._thread_state.last_gateway_metrics = {
                    "model_used": "gemini/gemini-2.5-flash-lite",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                }
                return (
                    [{"claim": text, "category": "other", "search_query": f"verify: {text}"}],
                    "gemini/gemini-2.5-flash-lite",
                )

            def fake_get_extract_metrics():
                return getattr(llm._thread_state, "last_gateway_metrics", None)

            def fake_verify(claim_item, tavily_client=None, use_cache=True, cache_path=None, dual_judge=False):
                seen_use_cache.append(use_cache)
                seen_dual_judge.append(dual_judge)
                c = claim_item["claim"]
                # Sleep in reverse order so out-of-order completion still preserves dataset order!
                if "Old rate" in c:
                    time.sleep(0.05)
                    return {
                        "status": "outdated",
                        "score": 15,
                        "confidence": "high",
                        "current_value": "5.50%",
                        "current_info": "Repo rate is 5.50%.",
                        "reason": "Lowered.",
                        "search_query": claim_item["search_query"],
                        "tavily_credits_used": 1.0,
                        "model": "openai/gpt-4o-mini",
                        "gateway_metrics": {
                            "model_used": "openai/gpt-4o-mini",
                            "prompt_tokens": 40,
                            "completion_tokens": 20,
                            "total_tokens": 60,
                        },
                        "sources": [{"title": "RBI", "url": "https://rbi.org.in", "source_type": "Primary"}],
                    }
                if "ChatGPT" in c:
                    time.sleep(0.01)
                    return {
                        "status": "fresh",
                        "score": 92,
                        "confidence": "high",
                        "current_value": "$20/mo",
                        "current_info": "Still $20/month.",
                        "reason": "Matches.",
                        "search_query": claim_item["search_query"],
                        "tavily_credits_used": 2.0,
                        "model": "openai/gpt-4o-mini",
                        "judge_telemetry": [
                            {"model_used": "openai/gpt-4o-mini", "prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40},
                            {"model_used": "anthropic/claude-3-5-sonnet", "prompt_tokens": 35, "completion_tokens": 15, "total_tokens": 50},
                        ],
                        "judge_models": ["openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet"],
                        "sources": [],
                    }
                if "React 18" in c:
                    # Simulate 1 miss (False Negative -> fresh instead of outdated)
                    return {
                        "status": "fresh",
                        "score": 80,
                        "confidence": "medium",
                        "current_info": "Misclassified.",
                        "reason": "Confused docs.",
                        "search_query": claim_item["search_query"],
                        "tavily_credits_used": 1.2,
                        "model": "openai/gpt-4o-mini",
                        "gateway_metrics": {"model_used": "openai/gpt-4o-mini", "prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                        "sources": [],
                    }
                return {
                    "status": "unverifiable",
                    "score": None,
                    "confidence": "low",
                    "current_info": "No sources.",
                    "reason": "None.",
                    "search_query": claim_item["search_query"],
                    "tavily_credits_used": 0.0,
                    "model": "openai/gpt-4o-mini",
                    "gateway_metrics": {"model_used": "openai/gpt-4o-mini", "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    "sources": [],
                }

            progress_events: list[tuple[int, int, int]] = []
            progress_lock = threading.Lock()

            def on_prog(done, total, row):
                with progress_lock:
                    progress_events.append((done, total, row["index"]))

            mock_tavily = MagicMock()
            with patch("benchmark.get_tavily_client", return_value=mock_tavily):
                with patch("benchmark.get_llm_config", return_value=("key", "http://localhost:8080/v1", ["m1"])):
                    with patch("benchmark._make_worker_tavily_client", return_value=(mock_tavily, False)):
                        with patch("benchmark.extract_claims", side_effect=fake_extract):
                            with patch("benchmark.get_extract_metrics", side_effect=fake_get_extract_metrics):
                                with patch("benchmark.verify_claim", side_effect=fake_verify):
                                    with patch("benchmark.is_use_bifrost", return_value=True):
                                        payload = benchmark.run_benchmark(
                                            dataset_path=ds_path,
                                            results_path=res_path,
                                            use_cache=False,
                                            dual_judge=True,
                                            max_workers=3,
                                            progress_callback=on_prog,
                                        )

            self.assertTrue(res_path.exists())
            self.assertEqual([e[0] for e in progress_events], [1, 2, 3, 4])
            self.assertEqual([r["index"] for r in payload["results"]], [1, 2, 3, 4])
            self.assertEqual(seen_use_cache, [False, False, False, False])
            self.assertEqual(seen_dual_judge, [True, True, True, True])
            self.assertTrue(payload["dual_judge"])
            self.assertFalse(payload["use_tavily_cache"])
            self.assertEqual(payload["routing_mode"], "bifrost")

            m = payload["metrics"]
            # Row 1: outdated==outdated (OK), Row 2: fresh==fresh (OK), Row 3: outdated!=fresh (MISS), Row 4: fresh!=unverifiable (MISS)
            self.assertEqual(m["correct"], 2)
            self.assertAlmostEqual(m["accuracy"], 0.5, places=4)
            # TP=1 (Row1), FP=0, FN=1 (Row3), TN=2 (Row2 & Row4)
            self.assertAlmostEqual(m["precision_stale"], 1.0, places=4)
            self.assertAlmostEqual(m["recall_stale"], 0.5, places=4)
            # Tokens: 4*15 (extract) + (60 + 90 + 30 + 15) (judge) = 255
            self.assertEqual(m["total_tokens"], 255)
            self.assertAlmostEqual(m["total_tavily_credits"], 4.2, places=2)
            self.assertIn("gemini/gemini-2.5-flash-lite", payload["models_used"])
            self.assertIn("anthropic/claude-3-5-sonnet", payload["models_used"])

            # Verify markdown & README injection
            md = benchmark.format_results_markdown(payload)
            self.assertIn("**50.0%** (2/4)", md)
            self.assertIn("100.0% (TP=1, FP=0)", md)
            self.assertIn("50.0% (TP=1, FN=1)", md)
            self.assertIn("✓ Correct", md)
            self.assertIn("✗ Incorrect", md)
            self.assertIn("N/A", md)

            updated_ok = benchmark.update_readme_benchmark_section(payload, readme_path=readme_path)
            self.assertTrue(updated_ok)
            updated_text = readme_path.read_text(encoding="utf-8")
            self.assertIn("# Header", updated_text)
            self.assertIn("## Footer", updated_text)
            self.assertIn("**50.0%** (2/4)", updated_text)
            self.assertNotIn("old placeholder", updated_text)

            no_markers = tmp_dir / "NO_MARKERS.md"
            no_markers.write_text("# Plain README\n", encoding="utf-8")
            self.assertFalse(benchmark.update_readme_benchmark_section(payload, readme_path=no_markers))

            # Outage guard: if every item errors out with 0 LLM tokens, existing results.json must be preserved!
            cached_before = res_path.read_text(encoding="utf-8")

            def failing_verify(claim_item, **kwargs):
                return {
                    "status": "unverifiable",
                    "score": None,
                    "confidence": "low",
                    "current_info": "Gateway offline.",
                    "reason": "Offline",
                    "tavily_credits_used": 0.0,
                    "sources": [],
                    "error": "Bifrost gateway is down or unreachable at http://localhost:8080/v1.",
                }

            with patch("benchmark.get_tavily_client", return_value=mock_tavily):
                with patch("benchmark.get_llm_config", return_value=("key", "http://localhost:8080/v1", ["m1"])):
                    with patch("benchmark._make_worker_tavily_client", return_value=(mock_tavily, False)):
                        with patch("benchmark.extract_claims", side_effect=RuntimeError("Gateway down")):
                            with patch("benchmark.get_extract_metrics", return_value=None):
                                with patch("benchmark.verify_claim", side_effect=failing_verify):
                                    with self.assertRaises(RuntimeError) as ctx:
                                        benchmark.run_benchmark(
                                            dataset_path=ds_path,
                                            results_path=res_path,
                                            max_workers=2,
                                        )
            self.assertIn("Bifrost gateway", str(ctx.exception))
            self.assertEqual(res_path.read_text(encoding="utf-8"), cached_before)

    def test_temp_cache_default_fallback_and_unwritable_no_crash(self):
        # 1. TEMP_RESULTS_PATH resolves inside OS temp directory.
        tmp_root = Path(tempfile.gettempdir()).resolve()
        self.assertTrue(benchmark.TEMP_RESULTS_PATH.resolve().is_relative_to(tmp_root))

        # 2. save_cached_results defaults to TEMP_RESULTS_PATH and load_cached_results prefers newer temp run.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            fake_temp_cache = tmp_dir / "temp_bench.json"
            fake_repo_cache = tmp_dir / "repo_bench.json"

            repo_payload = {"metrics": {"accuracy": 0.5}, "results": [{"index": 1, "source": "repo"}]}
            temp_payload = {"metrics": {"accuracy": 0.9}, "results": [{"index": 1, "source": "temp"}]}

            fake_repo_cache.write_text(json.dumps(repo_payload), encoding="utf-8")
            with patch.object(benchmark, "TEMP_RESULTS_PATH", fake_temp_cache), patch.object(
                benchmark, "RESULTS_PATH", fake_repo_cache
            ), patch.dict(os.environ, {"BENCHMARK_RESULTS_PATH": ""}, clear=False):
                saved = benchmark.save_cached_results(temp_payload)
                self.assertEqual(saved, fake_temp_cache)

                # Make temp file strictly newer than repo file and verify load_cached_results picks it.
                repo_mtime = fake_repo_cache.stat().st_mtime
                os.utime(fake_temp_cache, (repo_mtime + 10.0, repo_mtime + 10.0))
                loaded = benchmark.load_cached_results()
                self.assertEqual(loaded, temp_payload)

            # 3. Read-only / permission-denied destination never crashes save_cached_results.
            with patch("pathlib.Path.mkdir", side_effect=PermissionError("read-only fs")):
                res = benchmark.save_cached_results(temp_payload, results_path=tmp_dir / "blocked" / "results.json")
            self.assertIsNone(res)

            # 4. Unserializable payload also degrades safely to None.
            self.assertIsNone(benchmark.save_cached_results({"bad": {1, 2, 3}}, results_path=fake_temp_cache))


if __name__ == "__main__":
    unittest.main()
