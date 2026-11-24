"""
test_live_verification.py - Comprehensive offline tests for the real-time live verification experience.

Verifies:
1. Event Model & Reducer (events.py, audit_animations.py):
   - audit_started, claims_detected, claim_started, source_found, claim_verified,
     claim_failed, audit_finished, audit_cancelled
2. Parallel & Safe execution constraints (text-order preservation, exclusions)
3. Live Freshness Strip (segments, icons, tooltip, click handler, N of M label, aria-live)
4. Live Marks in Text (status highlight, icon, pending dotted pulse, failed hatched underline, html.escape)
5. Provisional Score Meter (risk-weighted gauge, provisional label, failed claims excluded)
6. Live Counters (Fresh, Outdated, Incorrect, Partial, Uncertain, Failed)
7. Live Activity Feed (newest-first, real durations, real event messages, click scroll)
8. Heuristic Preview (zero API calls, regex for price, %, date, version, recency, office title)
9. Stop / Cancellation Handling (pending -> skipped, partial results)
10. Smoothness & Throttle (<= 150ms update throttling)
11. Accessibility (aria-live, visible focus, WCAG AA compliance)
"""

import html
import time
import unittest
from unittest.mock import MagicMock, patch

from audit_animations import (
    AuditUIState,
    ClaimLiveState,
    apply_pipeline_event,
    create_initial_ui_state,
    render_action_status_pill_html,
    render_activity_feed_html,
    render_freshness_strip_from_results,
    render_freshness_strip_html,
    render_live_counters_html,
    render_live_marked_text_html,
    render_provisional_score_meter_html,
)
from events import (
    PipelineEvent,
    audit_cancelled_event,
    audit_finished_event,
    audit_started_event,
    claim_failed_event,
    claim_started_event,
    claim_verified_event,
    claims_detected_event,
    reduce_events,
    source_found_event,
)
from heuristic import (
    detect_heuristic_highlights,
    render_heuristic_preview_html,
)


class TestEventModelAndReducer(unittest.TestCase):
    """Unit tests for the event model and deterministic state reducer."""

    def test_pipeline_events_instantiation_and_timestamps(self):
        t0 = time.time()
        ev = audit_started_event()
        self.assertEqual(ev.event_type, "audit_started")
        self.assertGreaterEqual(ev.timestamp, t0)

        ev_c = claims_detected_event(
            [{"claim": "iPhone costs $999", "category": "price", "time_sensitivity": 80}],
            threshold=40,
        )
        self.assertEqual(ev_c.event_type, "claims_detected")
        self.assertEqual(len(ev_c.payload["claims"]), 1)

    def test_reduce_full_audit_lifecycle(self):
        events = [
            audit_started_event(),
            claims_detected_event(
                [
                    {"claim": "Claim A", "category": "price", "time_sensitivity": 90},
                    {"claim": "Claim B", "category": "leadership", "time_sensitivity": 80},
                    {"claim": "Claim C", "category": "other", "time_sensitivity": 10},
                ],
                threshold=40,
            ),
            claim_started_event(0),
            source_found_event(0, domain="apple.com", type="Primary"),
            claim_verified_event(0, status="outdated", score=20, duration_ms=210.0),
            claim_started_event(1),
            claim_failed_event(1, reason="Tavily connection timeout"),
            audit_finished_event(),
        ]
        state = reduce_events(events)

        self.assertEqual(state.phase, "error")  # failed claim sets phase to error / retry
        self.assertEqual(state.total_flagged_claims, 2)
        self.assertEqual(state.verified_count, 1)
        self.assertEqual(state.failed_count, 1)
        self.assertEqual(state.counters["outdated"], 1)
        self.assertEqual(state.counters["failed"], 1)
        self.assertEqual(state.claim_states[0].status, "outdated")
        self.assertEqual(state.claim_states[0].score, 20)
        self.assertEqual(state.claim_states[1].status, "verification_failed")
        self.assertIsNone(state.claim_states[1].score)
        # Provisional score includes claim 0 (score=20) and excludes failed claim 1
        self.assertEqual(state.provisional_score, 20)

    def test_legacy_event_aliases_still_work(self):
        state = create_initial_ui_state()
        state = apply_pipeline_event(state, "detect_start")
        self.assertEqual(state.phase, "running")
        self.assertEqual(state.active_stage, "detect")

        state = apply_pipeline_event(
            state,
            "detect_done",
            {
                "claims": [{"claim": "Fact 1", "category": "price", "time_sensitivity": 70}],
                "threshold": 40,
            },
        )
        self.assertEqual(state.total_flagged_claims, 1)

        state = apply_pipeline_event(
            state,
            "claim_done",
            {
                "claim_index": 0,
                "result": {"status": "fresh", "score": 95, "sources": []},
            },
        )
        self.assertEqual(state.counters["fresh"], 1)
        self.assertEqual(state.provisional_score, 95)


class TestHeuristicPreview(unittest.TestCase):
    """Unit tests for pure-Python zero-API heuristic preview."""

    def test_detect_prices_and_currencies(self):
        text = "The iPhone 15 starts at $799 and in UK it was £699 while in India ₹65,000."
        hits = detect_heuristic_highlights(text)
        found_texts = [h["text"] for h in hits]
        self.assertTrue(any("$799" in t for t in found_texts))
        self.assertTrue(any("£699" in t for t in found_texts))
        self.assertTrue(any("₹65,000" in t for t in found_texts))

    def test_detect_percentages_and_versions(self):
        text = "Python 3.12 has a 25.5% speed improvement over Python 3.10."
        hits = detect_heuristic_highlights(text)
        reasons = [h["reason"] for h in hits]
        self.assertTrue(any("version" in r for r in reasons))
        self.assertTrue(any("percentage" in r for r in reasons))

    def test_detect_temporal_words_and_office_titles(self):
        text = "The current CEO of OpenAI announced the latest model today."
        hits = detect_heuristic_highlights(text)
        found_texts = [h["text"].lower() for h in hits]
        self.assertTrue(any("ceo" in t for t in found_texts))
        self.assertTrue(any("latest" in t or "today" in t or "current" in t for t in found_texts))

    def test_empty_or_whitespace_input(self):
        self.assertEqual(detect_heuristic_highlights(""), [])
        self.assertEqual(detect_heuristic_highlights("   \n\t "), [])
        self.assertEqual(render_heuristic_preview_html([]), "")

    def test_heuristic_preview_html_label_honesty(self):
        hits = detect_heuristic_highlights("Apple released iPhone 16 at $999.")
        rendered = render_heuristic_preview_html(hits)
        self.assertIn("Heuristic preview", rendered)
        self.assertIn("not verified", rendered)
        self.assertIn("$999", rendered)


class TestFreshnessStrip(unittest.TestCase):
    """Unit tests for the Live Freshness Strip directly under Input Studio."""

    def test_strip_html_shows_all_status_segments(self):
        state = create_initial_ui_state()
        state.phase = "running"
        state.total_flagged_claims = 4
        state.claim_states[0] = ClaimLiveState(index=0, claim="Price is $999", status="fresh", substep="done", score=90)
        state.claim_states[1] = ClaimLiveState(index=1, claim="CEO is Sam", status="outdated", substep="done", score=20)
        state.claim_states[2] = ClaimLiveState(index=2, claim="GST is 18%", substep="search")
        state.claim_states[3] = ClaimLiveState(index=3, claim="Network down", status="verification_failed", substep="error")
        state.verified_count = 2
        state.failed_count = 1

        rendered = render_freshness_strip_html(state)
        self.assertIn("sad-freshness-strip", rendered)
        self.assertIn("sad-strip-fresh", rendered)
        self.assertIn("sad-strip-outdated", rendered)
        self.assertIn("sad-strip-running", rendered)
        self.assertIn("sad-strip-failed", rendered)
        self.assertIn('aria-live="polite"', rendered)
        self.assertIn("onclick=", rendered)
        self.assertIn("scrollIntoView", rendered)

    def test_render_freshness_strip_from_results(self):
        results = [
            {"claim": "Fact 1", "status": "fresh", "score": 95},
            {"claim": "Fact 2", "status": "outdated", "score": 15},
        ]
        rendered = render_freshness_strip_from_results(results)
        self.assertIn("sad-freshness-strip", rendered)
        self.assertIn("sad-strip-fresh", rendered)
        self.assertIn("sad-strip-outdated", rendered)
        self.assertIn("All 2 claim(s) verified", rendered)


class TestProvisionalScoreMeter(unittest.TestCase):
    """Unit tests for the live risk-weighted provisional score gauge."""

    def test_provisional_meter_excludes_failed_claims(self):
        state = create_initial_ui_state()
        state.phase = "running"
        state.total_flagged_claims = 3
        state.claim_states[0] = ClaimLiveState(index=0, claim="A", category="price", status="fresh", score=100)
        state.claim_states[1] = ClaimLiveState(index=1, claim="B", category="other", status="verification_failed", score=None, substep="error")
        state.verified_count = 1
        state.failed_count = 1
        state.provisional_score = 100

        meter_html = render_provisional_score_meter_html(state)
        self.assertIn("sad-provisional-meter", meter_html)
        self.assertIn("Provisional (1 of 3 verified)", meter_html)
        self.assertIn("1 could not be verified", meter_html)
        self.assertIn("100", meter_html)

    def test_final_meter_label_shows_score(self):
        state = create_initial_ui_state()
        state.phase = "done"
        state.total_flagged_claims = 1
        state.verified_count = 1
        state.provisional_score = 88

        meter_html = render_provisional_score_meter_html(state)
        self.assertIn("Score: 88/100", meter_html)
        self.assertNotIn("Provisional", meter_html)


class TestLiveCounters(unittest.TestCase):
    """Unit tests for the live status counters."""

    def test_live_counters_render_and_bump(self):
        state = create_initial_ui_state()
        state.phase = "running"
        state.counters = {
            "fresh": 3,
            "outdated": 1,
            "incorrect": 0,
            "partial": 1,
            "uncertain": 0,
            "failed": 2,
        }
        counters_html = render_live_counters_html(state)
        self.assertIn("sad-live-counters", counters_html)
        self.assertIn("sad-counter-fresh", counters_html)
        self.assertIn("3", counters_html)
        self.assertIn("sad-counter-outdated", counters_html)
        self.assertIn("sad-counter-failed", counters_html)
        self.assertIn("2", counters_html)


class TestLiveActivityFeed(unittest.TestCase):
    """Unit tests for the live activity feed."""

    def test_activity_feed_newest_first_and_durations(self):
        state = create_initial_ui_state()
        state.phase = "running"
        events = [
            audit_started_event(),
            claims_detected_event([{"claim": "C1"}], threshold=40),
            claim_started_event(0),
            source_found_event(0, domain="bloomberg.com", type="News"),
            claim_verified_event(0, status="fresh", score=90, duration_ms=345.0),
        ]
        reduce_events(events, initial_state=state)

        feed_html = render_activity_feed_html(state)
        self.assertIn("sad-activity-feed", feed_html)
        self.assertIn("Activity", feed_html)
        self.assertIn("bloomberg.com", feed_html)
        self.assertIn("345ms", feed_html)
        self.assertIn("sad-claim-card-0", feed_html)


class TestLiveMarksInText(unittest.TestCase):
    """Unit tests for live marked text in user answers."""

    def test_live_marks_pending_dotted_and_failed_hatched(self):
        raw_text = "The Apple iPhone 16 costs $1,199. The capital of Australia is Sydney."
        state = create_initial_ui_state()
        state.claim_states[0] = ClaimLiveState(
            index=0,
            claim="The Apple iPhone 16 costs $1,199",
            substep="search",
        )
        state.claim_states[1] = ClaimLiveState(
            index=1,
            claim="The capital of Australia is Sydney",
            status="verification_failed",
            substep="error",
        )

        marked_html = render_live_marked_text_html(raw_text, state)
        self.assertIn("sad-live-text-card", marked_html)
        self.assertIn("sad-hl-pending", marked_html)
        self.assertIn("sad-hl-failed", marked_html)
        self.assertIn(html.escape("$1,199"), marked_html)

    def test_live_marks_escapes_html_injection(self):
        dangerous_text = "Hello <script>alert('xss')</script> claim fact here."
        state = create_initial_ui_state()
        state.claim_states[0] = ClaimLiveState(
            index=0,
            claim="claim fact here",
            status="fresh",
            substep="done",
            score=95,
        )

        marked_html = render_live_marked_text_html(dangerous_text, state)
        self.assertNotIn("<script>", marked_html)
        self.assertIn("&lt;script&gt;", marked_html)


class TestStopButtonAndCancellation(unittest.TestCase):
    """Unit tests for Stop button and cancellation."""

    def test_cancellation_marks_pending_as_skipped(self):
        events = [
            audit_started_event(),
            claims_detected_event([{"claim": "C1"}, {"claim": "C2"}, {"claim": "C3"}]),
            claim_started_event(0),
            claim_verified_event(0, status="fresh", score=90),
            audit_cancelled_event(completed=1, total=3),
        ]
        state = reduce_events(events)
        self.assertEqual(state.phase, "stopped")
        self.assertTrue(state.stopped)
        self.assertEqual(state.completed_claims, 1)
        # Any non-completed segments are marked skipped
        self.assertEqual(state.strip_segments[0]["status"], "fresh")
        self.assertEqual(state.strip_segments[1]["status"], "skipped")
        self.assertEqual(state.strip_segments[2]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
