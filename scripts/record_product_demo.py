import os
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

def smooth_move(page, start_pos, target_pos, steps=25, total_duration=0.7):
    """Interpolate smooth cursor movement with natural easing."""
    sx, sy = start_pos
    tx, ty = target_pos
    step_duration = total_duration / max(steps, 1)
    for i in range(1, steps + 1):
        t = i / steps
        ease = t * t * (3 - 2 * t)
        cx = sx + (tx - sx) * ease
        cy = sy + (ty - sy) * ease
        page.mouse.move(cx, cy)
        time.sleep(step_duration)
    return target_pos

def smooth_scroll(page, delta_y, steps=20, delay=0.03):
    """Scroll smoothly down or up in small increments."""
    dy = delta_y / max(steps, 1)
    for _ in range(steps):
        page.mouse.wheel(0, dy)
        time.sleep(delay)

def main():
    out_dir = Path("unveo-out")
    raw_dir = out_dir / "raw_record"
    raw_dir.mkdir(parents=True, exist_ok=True)
    final_video_path = out_dir / "demo-video-80s.mp4"

    print("[1/8] Starting Playwright video recording at 1920x1080...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--start-maximized", "--disable-blink-features=AutomationControlled"]
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=str(raw_dir),
            record_video_size={"width": 1920, "height": 1080},
        )
        page = context.new_page()
        page.set_default_timeout(60000)

        start_time = time.time()
        cursor = (960, 540)
        page.mouse.move(*cursor)

        # -------------------------------------------------------------
        # STEP 1: Open app & click Guided Demo
        # -------------------------------------------------------------
        print("[2/8] Step 1: Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle")
        page.locator('text="Input Studio"').first.wait_for(timeout=30000)
        time.sleep(2.0)

        print("[2/8] Step 1: Clicking Guided Demo...")
        guided_btn = page.locator('button:has-text("Guided Demo")').first
        guided_btn.wait_for(state="visible", timeout=10000)
        box = guided_btn.bounding_box()
        if box:
            cursor = smooth_move(page, cursor, (box["x"] + box["width"]/2, box["y"] + box["height"]/2))
        guided_btn.click()
        time.sleep(2.5)

        # -------------------------------------------------------------
        # STEP 2: Enter exact text & click Check Freshness
        # -------------------------------------------------------------
        print("[3/8] Step 2: Entering query into Paste an answer...")
        text_to_paste = (
            "GPT-4 is OpenAI's most advanced model. "
            "India's GST has four slabs: 5%, 12%, 18% and 28%. "
            "The latest iPhone is the iPhone 17."
        )
        textarea = page.locator('textarea').first
        textarea.wait_for(state="visible", timeout=10000)
        t_box = textarea.bounding_box()
        if t_box:
            cursor = smooth_move(page, cursor, (t_box["x"] + t_box["width"]/2, t_box["y"] + t_box["height"]/2))
        textarea.click()
        textarea.fill(text_to_paste)
        time.sleep(2.0)

        print("[3/8] Step 2: Clicking Check Freshness...")
        check_btn = page.locator('button:has-text("Check Freshness")').first
        check_btn.wait_for(state="visible", timeout=10000)
        c_box = check_btn.bounding_box()
        if c_box:
            cursor = smooth_move(page, cursor, (c_box["x"] + c_box["width"]/2, c_box["y"] + c_box["height"]/2))
        check_btn.click()

        # -------------------------------------------------------------
        # STEP 3: Wait for 5-step progress & view claim results
        # -------------------------------------------------------------
        print("[4/8] Step 3: Waiting for 5-step progress and live verification results...")
        time.sleep(3.0)
        # Wait until progress finishes and results header appears
        page.locator('text="Answer and summary"').first.wait_for(state="visible", timeout=60000)
        time.sleep(2.5)

        print("[4/8] Step 3: Scrolling slowly across annotated answer...")
        smooth_scroll(page, 280, steps=20)
        time.sleep(2.0)

        # Hover over one highlighted claim
        hl_span = page.locator('[class*="sad-hl"], [class*="sad-phrase"], mark').first
        if hl_span.count() > 0:
            h_box = hl_span.bounding_box()
            if h_box:
                cursor = smooth_move(page, cursor, (h_box["x"] + h_box["width"]/2, h_box["y"] + h_box["height"]/2))
                time.sleep(2.5)

        print("[4/8] Step 3: Scrolling to verification summary & claim cards...")
        smooth_scroll(page, 380, steps=20)
        time.sleep(2.5)

        # Open one claim card and pause on Claimed vs Actually (today)
        expander = page.locator('[data-testid="stExpander"]').first
        if expander.count() > 0:
            e_box = expander.bounding_box()
            if e_box:
                cursor = smooth_move(page, cursor, (e_box["x"] + 120, e_box["y"] + 20))
                toggle_btn = expander.locator('summary, button, [data-testid="stExpanderToggleIcon"]').first
                if toggle_btn.count() > 0:
                    toggle_btn.click()
                else:
                    expander.click()
                time.sleep(1.5)

        smooth_scroll(page, 300, steps=20)
        print("[4/8] Step 3: Pausing on Claimed vs Actually (today) and primary source chips...")
        time.sleep(3.5)

        # -------------------------------------------------------------
        # STEP 4: Open corrected answer / diff view
        # -------------------------------------------------------------
        print("[5/8] Step 4: Viewing corrected answer / diff view...")
        smooth_scroll(page, 450, steps=25)
        diff_view = page.locator('text="Original answer", text="Corrected answer"').first
        if diff_view.count() > 0:
            dv_box = diff_view.bounding_box()
            if dv_box:
                cursor = smooth_move(page, cursor, (dv_box["x"] + 100, dv_box["y"] + 20))
        time.sleep(3.5)

        # -------------------------------------------------------------
        # STEP 5: Gateway panel & Outage simulation
        # -------------------------------------------------------------
        print("[6/8] Step 5: Gateway panel & Outage simulation...")
        sidebar = page.locator('[data-testid="stSidebar"]').first
        if sidebar.count() > 0:
            gw_header = sidebar.locator('text="Gateway panel"').first
            if gw_header.count() > 0:
                gw_header.scroll_into_view_if_needed()
                gw_box = gw_header.bounding_box()
                if gw_box:
                    cursor = smooth_move(page, cursor, (gw_box["x"] + 80, gw_box["y"] + 20))
                time.sleep(1.0)
            
            # Turn on the outage simulation toggle
            outage_toggle = sidebar.locator('label:has-text("Simulate provider outage")').first
            if outage_toggle.count() > 0:
                ot_box = outage_toggle.bounding_box()
                if ot_box:
                    cursor = smooth_move(page, cursor, (ot_box["x"] + 25, ot_box["y"] + ot_box["height"]/2))
                outage_toggle.click(force=True)
                time.sleep(1.5)

        # Re-run one claim simulation
        why_bifrost_btn = page.locator('button:has-text("Why Bifrost")').first
        if why_bifrost_btn.count() > 0:
            why_bifrost_btn.scroll_into_view_if_needed()
            time.sleep(1.0)
            wb_box = why_bifrost_btn.bounding_box()
            if wb_box:
                cursor = smooth_move(page, cursor, (wb_box["x"] + wb_box["width"]/2, wb_box["y"] + wb_box["height"]/2))
            why_bifrost_btn.click()
            time.sleep(2.0)

            # Wait for simulation failover results container
            sim_header = page.locator('text="Why Bifrost? Upstream Outage Simulation & Failover"').first
            if sim_header.count() > 0:
                sim_header.wait_for(state="visible", timeout=15000)
                smooth_scroll(page, 200, steps=15)
            
            # Pause on Fallback chip showing automatic model switch
            fb_badge = page.locator('text="Responded:"').first
            if fb_badge.count() > 0:
                fb_box = fb_badge.bounding_box()
                if fb_box:
                    cursor = smooth_move(page, cursor, (fb_box["x"] + 80, fb_box["y"] + 15))
            print("[6/8] Step 5: Pausing on Fallback chip & automatic model switch...")
            time.sleep(3.5)

            # Turn it off
            why_bifrost_btn.scroll_into_view_if_needed()
            why_bifrost_btn.click()
            time.sleep(1.0)
        else:
            rerun_btn = page.locator('button:has-text("Re-run claim")').first
            if rerun_btn.count() > 0:
                rerun_btn.scroll_into_view_if_needed()
                time.sleep(1.0)
                rb_box = rerun_btn.bounding_box()
                if rb_box:
                    cursor = smooth_move(page, cursor, (rb_box["x"] + rb_box["width"]/2, rb_box["y"] + rb_box["height"]/2))
                rerun_btn.click()
                time.sleep(2.5)
                fb_chip = page.locator('[class*="sad-fallback"], text="Fallback"').first
                if fb_chip.count() > 0:
                    fb_box = fb_chip.bounding_box()
                    if fb_box:
                        cursor = smooth_move(page, cursor, (fb_box["x"] + 50, fb_box["y"] + 10))
                print("[6/8] Step 5: Pausing on Fallback chip & automatic model switch...")
                time.sleep(3.5)

        if sidebar.count() > 0:
            outage_toggle = sidebar.locator('label:has-text("Simulate provider outage")').first
            if outage_toggle.count() > 0:
                ot_box = outage_toggle.bounding_box()
                if ot_box:
                    cursor = smooth_move(page, cursor, (ot_box["x"] + 25, ot_box["y"] + ot_box["height"]/2))
                outage_toggle.click(force=True)
                time.sleep(1.5)

        # -------------------------------------------------------------
        # STEP 6: Open Benchmark tab and show metrics
        # -------------------------------------------------------------
        print("[7/8] Step 6: Opening Benchmark / Agent Lab tab...")
        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        time.sleep(1.5)

        lab_tab = page.locator('[data-testid="stTab"]:has-text("AGENT LAB"), [role="tab"]:has-text("AGENT LAB")').first
        lab_tab.wait_for(state="visible", timeout=10000)
        l_box = lab_tab.bounding_box()
        if l_box:
            cursor = smooth_move(page, cursor, (l_box["x"] + l_box["width"]/2, l_box["y"] + l_box["height"]/2))
        lab_tab.click()
        time.sleep(2.5)

        # Switch to benchmark sub-view if present
        bench_seg = page.locator('text="Ground Truth Benchmark"').first
        if bench_seg.count() > 0:
            bench_seg.click()
            time.sleep(2.0)

        smooth_scroll(page, 320, steps=20)
        print("[7/8] Step 6: Pausing on benchmark accuracy, latency, and cost metrics...")
        time.sleep(4.0)

        # -------------------------------------------------------------
        # STEP 7: Finish on Audit tab with title visible
        # -------------------------------------------------------------
        print("[8/8] Step 7: Returning to AUDIT tab and showing title...")
        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        time.sleep(1.0)

        audit_tab = page.locator('[data-testid="stTab"]:has-text("AUDIT"), [role="tab"]:has-text("AUDIT")').first
        audit_tab.wait_for(state="visible", timeout=10000)
        a_box = audit_tab.bounding_box()
        if a_box:
            cursor = smooth_move(page, cursor, (a_box["x"] + a_box["width"]/2, a_box["y"] + a_box["height"]/2))
        audit_tab.click()
        time.sleep(2.0)

        # Scroll to absolute top so title and Input Studio are prominently visible
        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        time.sleep(1.0)
        cursor = smooth_move(page, cursor, (960, 160))

        # Precision 80-second duration pad
        elapsed = time.time() - start_time
        remaining = 80.0 - elapsed
        print(f"Elapsed: {elapsed:.1f}s | Target: 80s | Remaining padding: {remaining:.1f}s")
        if remaining > 0:
            time.sleep(remaining)

        # Finalize and close video
        video = page.video
        context.close()
        browser.close()

        if video:
            saved_video_path = Path(video.path())
            print(f"Raw Playwright video recorded to: {saved_video_path}")
            
            # Re-encode to polished 1920x1080 MP4 with ffmpeg
            ffmpeg_exe = r"C:\Users\PUBLIC.SURAJ-TOMAR\.unveo\venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v9.0.2.exe"
            if Path(ffmpeg_exe).is_file():
                import subprocess
                print(f"Re-encoding to polished 1920x1080 MP4: {final_video_path}...")
                subprocess.run([
                    ffmpeg_exe, "-y", "-i", str(saved_video_path),
                    "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-r", "30",
                    "-pix_fmt", "yuv420p", str(final_video_path)
                ], check=True)
                print(f"Successfully generated: {final_video_path}")
                return str(final_video_path)
            return str(saved_video_path)

    return None

if __name__ == "__main__":
    vid = main()
    print("FINISHED_RECORDING_PATH=" + str(vid))
