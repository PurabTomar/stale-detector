"""Offline unit tests for the Check Freshness real-event animation system.

Covers:
1. Event-to-UI-state mapping: idle, running (detect, search, extract, judge, score),
   done (success), error (detect/verify/score/report), and stopped (partial results).
2. Real event emission and Stop cancellation in verifier.verify_claims().
3. Pure HTML renderers: button status pill, scan overlay, claim discovery chips,
   live verify skeleton board + Sources read counter, stepper SVG draw-on & L-to-R
   connector fill, SVG score ring draw-on & stagger reveal, and XSS escaping.
4. CSS constraints: balanced braces, non-loop durations < 600ms, reduced-motion
   static progress bar fallback, no confetti or audio.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import threading
import unittest
from unittest.mock import MagicMock, patch

import app
from audit_animations import (
    AuditUIState,
    apply_pipeline_event,
    create_initial_ui_state,
    render_action_status_pill_html,
    render_claim_discovery_chips_html,
    render_reduced_motion_progress_html,
    render_scan_effect_html,
    render_stepper_check_svg,
    render_verify_live_board_html,
)
import verifier


class TestEventToUIStateMapping(unittest.TestCase):
    """Test deterministic real pipeline event -> UI state transitions."""

    def test_full_running_to_done_lifecycle(self):
        state = create_initial_ui_state()
        self.assertEqual(state.phase, "idle")
        self.assertEqual(state.button_state, "idle")
        self.assertFalse(state.button_disabled)
        self.assertEqual(list(state.stepper.values()), ["pending"] * 5)

        # 1. detect_start
        apply_pipeline_event(state, "detect_start")
        self.assertEqual(state.phase, "running")
        self.assertEqual(state.button_state, "loading")
        self.assertEqual(state.button_label, "Detecting claims")
        self.assertTrue(state.button_disabled)
        self.assertTrue(state.scan_active)
        self.assertTrue(state.input_glow)
        self.assertEqual(state.stepper["input"], "done")
        self.assertEqual(state.stepper["detect"], "running")
        self.assertEqual(state.stepper["verify"], "pending")

        # 2. detect_done (1 flagged, 1 stable)
        claims = [
            {"claim": "iPhone 16 starts at $799.", "category": "price", "time_sensitivity": 85},
            {"claim": "Water boils at 100C.", "category": "other", "time_sensitivity": 10},
        ]
        apply_pipeline_event(state, "detect_done", {"claims": claims, "threshold": 40})
        self.assertFalse(state.scan_active)
        self.assertFalse(state.input_glow)
        self.assertEqual(state.total_flagged_claims, 1)
        self.assertEqual(state.stepper["detect"], "done")
        self.assertEqual(state.stepper["verify"], "running")
        self.assertEqual(state.button_label, "Searching live sources")

        # 3. claim_substep: search -> extract (with real arriving sources) -> judge
        apply_pipeline_event(state, "claim_substep", {"claim_index": 0, "substep": "search"})
        self.assertEqual(state.claim_states[0].substep, "search")
        self.assertEqual(state.button_label, "Searching live sources")

        apply_pipeline_event(
            state,
            "claim_substep",
            {
                "claim_index": 0,
                "substep": "extract",
                "sources": [
                    {"url": "https://www.apple.com/iphone/", "domain": "apple.com", "source_type": "Primary"},
                    {"url": "https://www.reuters.com/tech/iphone", "domain": "reuters.com", "source_type": "News"},
                ],
            },
        )
        self.assertEqual(state.claim_states[0].substep, "extract")
        self.assertEqual(state.sources_read, 2)
        self.assertEqual(len(state.claim_states[0].sources), 2)

        apply_pipeline_event(state, "claim_substep", {"claim_index": 0, "substep": "judge"})
        self.assertEqual(state.claim_states[0].substep, "judge")
        self.assertEqual(state.button_label, "Judging")

        # 4. claim_done -> automatically enters Scoring when last claim finishes
        apply_pipeline_event(
            state,
            "claim_done",
            {
                "claim_index": 0,
                "result": {
                    "claim": "iPhone 16 starts at $799.",
                    "status": "fresh",
                    "score": 94,
                    "sources": [
                        {"url": "https://www.apple.com/iphone/", "domain": "apple.com", "source_type": "Primary"},
                        {"url": "https://www.theverge.com/iphone", "domain": "theverge.com", "source_type": "News"},
                    ],
                },
            },
        )
        self.assertEqual(state.completed_claims, 1)
        self.assertEqual(state.sources_read, 3)  # apple.com deduped, theverge.com added
        self.assertEqual(state.claim_states[0].substep, "done")
        self.assertEqual(state.button_label, "Scoring")
        self.assertEqual(state.stepper["verify"], "done")
        self.assertEqual(state.stepper["score"], "running")

        # 5. complete
        apply_pipeline_event(state, "complete", {"results": [{"status": "fresh", "score": 94}]})
        self.assertEqual(state.phase, "done")
        self.assertEqual(state.button_state, "success")
        self.assertFalse(state.button_disabled)
        self.assertTrue(state.reveal_ready)
        self.assertEqual(state.progress_ratio, 1.0)
        self.assertEqual(list(state.stepper.values()), ["done"] * 5)

    def test_error_mapping_sets_retry_and_keeps_later_steps_pending(self):
        state = create_initial_ui_state()
        apply_pipeline_event(state, "detect_start")
        apply_pipeline_event(state, "error", {"step": "detect", "reason": "Connection refused"})

        self.assertEqual(state.phase, "error")
        self.assertEqual(state.button_state, "error")
        self.assertEqual(state.button_label, "Retry")
        self.assertFalse(state.button_disabled)
        self.assertFalse(state.scan_active)
        self.assertEqual(
            state.stepper,
            {
                "input": "done",
                "detect": "error",
                "verify": "pending",
                "score": "pending",
                "report": "pending",
            },
        )

        # Complete with a verification_failed claim also maps to error on verify
        state2 = create_initial_ui_state()
        apply_pipeline_event(
            state2,
            "complete",
            {"results": [{"status": "verification_failed", "score": None}]},
        )
        self.assertEqual(state2.phase, "error")
        self.assertEqual(state2.button_state, "error")
        self.assertEqual(state2.button_label, "Retry")
        self.assertEqual(state2.stepper["verify"], "error")
        self.assertEqual(state2.stepper["score"], "pending")
        self.assertEqual(state2.stepper["report"], "pending")

    def test_stopped_state_preserves_partial_progress(self):
        state = create_initial_ui_state()
        apply_pipeline_event(state, "detect_start")
        apply_pipeline_event(
            state,
            "detect_done",
            {
                "claims": [
                    {"claim": "Claim 1", "category": "price", "time_sensitivity": 80},
                    {"claim": "Claim 2", "category": "version", "time_sensitivity": 80},
                ],
                "threshold": 40,
            },
        )
        apply_pipeline_event(
            state,
            "claim_done",
            {"claim_index": 0, "result": {"claim": "Claim 1", "status": "fresh", "score": 90, "sources": []}},
        )
        apply_pipeline_event(state, "stopped", {"completed": 1, "total": 2})

        self.assertEqual(state.phase, "stopped")
        self.assertTrue(state.stopped)
        self.assertFalse(state.button_disabled)
        self.assertEqual(state.completed_claims, 1)
        self.assertEqual(state.total_flagged_claims, 2)
        self.assertTrue(state.reveal_ready)
        self.assertEqual(state.stepper["verify"], "done")
        self.assertEqual(state.stepper["score"], "pending")


class TestVerifierLiveEventsAndStopCancellation(unittest.TestCase):
    """Test real event callbacks and Stop cancellation in verifier.verify_claims()."""

    @patch("verifier._cache_set")
    @patch("verifier._cache_get", return_value=None)
    @patch("verifier._invoke_call_llm")
    @patch("verifier.get_tavily_client")
    def test_verify_claims_emits_real_substeps_and_cancels_cleanly(
        self, mock_get_client, mock_llm, _mock_cache, _mock_store
    ):
        client = MagicMock()
        client.search.return_value = {
            "results": [
                {
                    "url": "https://openai.com/api/pricing/",
                    "title": "OpenAI Pricing",
                    "content": "GPT-4o costs $2.50 per 1M input tokens.",
                    "score": 0.95,
                }
            ]
        }
        mock_get_client.return_value = client
        judge_json = json.dumps({
            "evidence_stance": "supports",
            "supporting_quote": "GPT-4o costs $2.50 per 1M input tokens.",
            "status": "fresh",
            "confidence": "high",
            "was_ever_true": "yes",
            "current_info": "GPT-4o costs $2.50 per 1M input tokens.",
            "reason": "Confirmed by OpenAI official pricing.",
            "agreeing_sources": 1,
            "conflicting_sources": False,
            "source_evaluations": [
                {"url": "https://openai.com/api/pricing/", "stance": "support", "quote": "GPT-4o costs $2.50", "source_type": "primary"}
            ],
        })
        mock_llm.return_value = (
            judge_json,
            {"model_used": "gemini-3.1-flash-lite", "latency_ms": 100.0, "total_tokens": 90, "usage_returned": True},
        )

        # 1. Verify real substeps emitted in order on 1 claim
        emitted: list[tuple[int, str, dict]] = []
        claims = [
            {"claim": "GPT-4o costs $2.50 per 1M input tokens.", "category": "pricing", "search_query": "GPT-4o pricing"},
            {"claim": "Claude 3.5 Sonnet costs $3 per 1M input tokens.", "category": "pricing", "search_query": "Claude pricing"},
        ]

        cancel_ev = threading.Event()

        def on_ev(c_idx: int, substep: str, payload: dict) -> None:
            emitted.append((c_idx, substep, payload))

        def on_prog(done_cnt: int, total_cnt: int, _res: dict) -> None:
            # Stop right after the 1st claim completes!
            if done_cnt == 1:
                cancel_ev.set()

        partial_results = verifier.verify_claims(
            claims,
            progress_callback=on_prog,
            event_callback=on_ev,
            cancel_event=cancel_ev,
            max_workers=1,  # Deterministic sequential execution for offline cancellation test
        )

        # Claim 0 completed cleanly; Claim 1 was cancelled cleanly -> 1 partial result returned
        self.assertEqual(len(partial_results), 1)
        self.assertEqual(partial_results[0]["status"], "fresh")
        substeps_claim0 = [sub for idx, sub, _ in emitted if idx == 0]
        self.assertIn("search", substeps_claim0)
        self.assertIn("extract", substeps_claim0)
        self.assertIn("judge", substeps_claim0)
        self.assertIn("score", substeps_claim0)
        self.assertIn("done", substeps_claim0)
        extract_events = [p for idx, sub, p in emitted if idx == 0 and sub == "extract"]
        self.assertTrue(extract_events and len(extract_events[0].get("sources", [])) >= 1)


class TestAnimationHtmlRenderersAndCssGuards(unittest.TestCase):
    """Test HTML renderers, XSS escaping, reduced motion, and CSS performance rules."""

    def test_renderers_escape_xss_and_render_required_classes(self):
        # A. Button status pills
        st_load = apply_pipeline_event(create_initial_ui_state(), "detect_start")
        html_load = render_action_status_pill_html(st_load)
        self.assertIn("sad-btn-loading", html_load)
        self.assertIn("sad-btn-spinner", html_load)
        self.assertIn("Detecting claims", html_load)
        self.assertIn("sad-reduced-progress", html_load)

        st_ok = apply_pipeline_event(create_initial_ui_state(), "complete", {"results": [{"status": "fresh", "score": 90}]})
        html_ok = render_action_status_pill_html(st_ok)
        self.assertIn("sad-btn-success", html_ok)
        self.assertIn("sad-check-path", html_ok)

        st_err = apply_pipeline_event(create_initial_ui_state(), "error", {"step": "detect", "reason": "<script>alert(1)</script>"})
        html_err = render_action_status_pill_html(st_err)
        self.assertIn("sad-btn-error", html_err)
        self.assertIn("Retry", html_err)
        self.assertNotIn("<script>", html_err)

        # B. Scan effect
        scan_html = render_scan_effect_html("Check <img src=x onerror=alert(1)> text", True)
        self.assertIn("sad-scan-beam", scan_html)
        self.assertNotIn("<img", scan_html)
        self.assertEqual(render_scan_effect_html("text", False), "")

        # C. Claim discovery chips (amber flagged vs grey stable + scaleX bar)
        disc_html = render_claim_discovery_chips_html(
            [
                {"claim": "Flagged <script>x</script>", "category": "price", "time_sensitivity": 90},
                {"claim": "Stable fact", "category": "other", "time_sensitivity": 15},
            ],
            threshold=40,
            category_icons={"price": "💰 ", "other": "📌 "},
        )
        self.assertIn("sad-discovery-chip", disc_html)
        self.assertIn("sad-disc-dot-amber", disc_html)
        self.assertIn("sad-disc-dot-grey", disc_html)
        self.assertIn("--sad-bar-scale:", disc_html)
        self.assertNotIn("<script>", disc_html)

        # E. Verify live board (skeleton + Search/Extract/Judge + Sources read: N)
        st_ver = create_initial_ui_state()
        apply_pipeline_event(st_ver, "detect_done", {"claims": [{"claim": "Price is $10", "category": "price", "time_sensitivity": 80}], "threshold": 40})
        apply_pipeline_event(
            st_ver,
            "claim_substep",
            {
                "claim_index": 0,
                "substep": "extract",
                "sources": [{"url": "https://apple.com", "domain": "apple.com", "source_type": "Primary"}],
            },
        )
        board_html = render_verify_live_board_html(st_ver)
        self.assertIn("sad-verify-skel-card", board_html)
        self.assertIn("Search", board_html)
        self.assertIn("Extract", board_html)
        self.assertIn("Judge", board_html)
        self.assertIn("Sources read: <strong>1</strong>", board_html)
        self.assertIn("sad-live-src-chip", board_html)

        # D. Stepper with draw-on SVG check & L-to-R connector track
        with patch("streamlit.session_state", {}):
            stepper_html = app.render_stepper_html("pasted", has_results=False, has_input=True, ui_state=st_ver)
        self.assertIn("sad-pipe-check-svg", stepper_html)
        self.assertIn("sad-pipe-connector-track pipe-connector-done", stepper_html)
        self.assertIn("sad-pipe-step pipe-running", stepper_html)

        # F. SVG score ring draw-on & data-target-score
        ring_html = app.render_score_ring_html(88, ["gemini-3.1-flash-lite"], results=[{"status": "fresh", "score": 88}])
        self.assertIn("sad-ring-svg", ring_html)
        self.assertIn("sad-ring-progress-circle", ring_html)
        self.assertIn('data-target-score="88"', ring_html)
        self.assertIn("sad-reveal-stagger", ring_html)
        self.assertIn("sad-pill-pop", ring_html)

    def test_css_braces_reduced_motion_and_duration_budget(self):
        css_path = Path(__file__).resolve().parent / "assets" / "style.css"
        css = css_path.read_text(encoding="utf-8")

        self.assertEqual(css.count("{"), css.count("}"), "Unbalanced CSS braces in assets/style.css")
        self.assertIn("@media (prefers-reduced-motion: reduce)", css)
        self.assertIn(".sad-reduced-progress", css)
        self.assertIn("sadStaleWipe", css)
        self.assertIn("sadFreshGlowOnce", css)
        self.assertIn("sadRingStrokeDraw", css)
        self.assertNotIn("confetti", css.lower())

        # Every non-infinite animation in our new animation section must be < 600ms
        marker = "Check Freshness Honest Pipeline Animations"
        self.assertIn(marker, css)
        anim_section = css.split(marker, 1)[1]
        for line in anim_section.splitlines():
            if "animation:" in line and "infinite" not in line and "none" not in line:
                m = re.search(r"(\d+(?:\.\d+)?)\s*(ms|s)\b", line)
                if m:
                    val = float(m.group(1))
                    ms = val if m.group(2) == "ms" else val * 1000.0
                    self.assertLess(ms, 600.0, f"Non-loop animation exceeds 600ms budget: {line.strip()}")


if __name__ == "__main__":
    unittest.main()
