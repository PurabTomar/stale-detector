# Stale-Answer Detector & Agent Lab — 3-Minute Demo Video Script

**Target Duration:** `3:00` (`180 seconds`)  
**Resolution:** `1920x1080` (browser at `1440px` zoom 100%, dark mode ON)

---

## Pre-Recording Setup Checklist

1. Start local Bifrost AI Gateway on port `8080` (`http://localhost:8080`) with two configured providers (e.g., Gemini + fallback) and confirm **Gateway healthy** in the sidebar.
   - *(If recording without live keys, note clearly on screen that **Demo mode** is enabled; however, the outage fallback + Bifrost dashboard segment requires a running local Bifrost instance on `http://localhost:8080` — mark `TODO` if Bifrost is not yet running.)*
2. Start the Streamlit app: `streamlit run app.py --server.headless true` (open `http://localhost:8501`).
3. Start the FastAPI Guard API in a second terminal: `uvicorn api:app --host 0.0.0.0 --port 8000` (open `http://localhost:8000/docs` in Tab 2 of browser).
4. Open the Bifrost dashboard logs (`http://localhost:8080`) in Tab 3 of browser.
5. **Security check:** Never open `.env` or display any API key on screen. Close notifications and clear previous session history via **Clear** so the workspace starts clean.

---

## Timestamped Script & Exact Clicks

### `0:00 – 0:20` | Hook & Problem Setup (20s)
- **Screen:** Browser Tab 1 (`http://localhost:8501`) on **AUDIT** tab. Show top header (**Stale-Answer Detector**, **Quota: Live**, **Routed via Bifrost**, 5-stage stepper all pending, always-open **Input Studio**).
- **Exact Clicks:** None yet — slow cursor sweep over header chips, stepper, and Input Studio.
- **Narration:**
  > "LLM answers sound confident even months after prices, versions, tax slabs, or leadership change. This is **Stale-Answer Detector** and **Agent Lab**—built with **Tavily** for primary-source grounding and **Bifrost by Maxim AI** for multi-model routing and failover."

---

### `0:20 – 1:05` | Live Audit Animation, Verification & Cited Corrections (45s)
- **Exact Clicks:**
  1. In **Input Studio**, click the **`iPhone`** or **`India`** example chip (or paste a 3-sentence answer containing one fresh claim, one outdated/incorrect claim, and one unverified claim).
  2. Click **`Check Freshness`** (primary button).
  3. **Pause to show real-event animations:**
     - Button immediately disables with shimmering gradient + spinner + real step label (`Detecting claims` → `Searching live sources` → `Judging` → `Scoring`), while the **Stop** button appears next to it.
     - Soft gradient scan beam sweeps across the text during detection.
     - Detected claims pop in as chips with category icons, amber/grey dots, and animated time-sensitivity bars.
     - Stepper moves from **02 Detect** (green SVG check + L-to-R connector fill) to **03 Verify** (pulsing indigo).
     - Live Verify skeleton cards show `Search → Extract → Judge` substeps, arriving source chips, and real `Sources read: N`.
  4. Wait for completion: staggered fade-up of **Verification Summary** (SVG draw-on score ring + count-up), inline stale highlight wipe in **LLM Answer**, and claim cards.
  5. Scroll to a claim card and point to its deduplicated `PRIMARY` / `NEWS` domain chip (`+N pages`), `N of M independent sources agree`, and **Evidence snippets**.
  6. Scroll to **Generate corrected answer** and click **`Generate corrected answer`**.
  7. Point to the side-by-side red/green word diff and the **Cited corrected answer & primary sources** card (`[1]`, `[2]` clickable citations + `Sources` list + `[unverified]` marker on untouched unverified claims).
  8. Hover over **`Download report (Markdown)`** and **`Download HTML`** export buttons.
- **Narration:**
  > "I paste an AI answer and click **Check Freshness**. Every animation is driven by real pipeline events—no fake progress. First it scans for claims and flags time-sensitive ones in amber; next, Tavily runs primary-first searches and deep extraction while `Sources read` counts up live. In the results, stale phrases wipe across the text, every domain is deduplicated with one authority badge, and clicking **Generate corrected answer** replaces only verified stale facts with numbered `[1]`, `[2]` citations to primary sources—leaving unverified claims untouched and marked `[unverified]`, ready for Markdown or HTML export."

---

### `1:05 – 1:50` | Bifrost Outage Fallback & Gateway Logs (45s)
- **Exact Clicks:**
  1. Open the left **Sidebar** and point to the **Bifrost Gateway** panel (**Gateway healthy**, active `Extract` and `Judge` task models, live request count, latency, and token totals).
  2. Toggle **`Simulate provider outage`** **ON** in the sidebar.
  3. Click **`Check Freshness`** (or **`Ask & Audit`**) to trigger a live request through Bifrost with the primary model forced to fail.
  4. Point directly to the **`Fallback used: A → B`** badge in both the **Verification Summary** and the sidebar **Bifrost Gateway** panel.
  5. Click the **`Bifrost dashboard logs`** link (or switch to Browser Tab 3 at `http://localhost:8080`).
  6. Click **Logs** in the Bifrost dashboard and open the latest request entry showing the primary model failure, automatic failover to the backup model, latency, and token usage.
  7. Switch back to Browser Tab 1 and toggle **`Simulate provider outage`** **OFF**.
- **Narration:**
  > "Now let's test resilience. In the sidebar, I toggle **Simulate provider outage** ON and run a check. Instead of crashing, **Bifrost** automatically fails over to our backup provider—you can see the `Fallback used` route badge right in the summary and sidebar telemetry. Switching to the **Bifrost dashboard logs** at `localhost:8080`, we see the exact failover event, model route, latency, and token counts recorded live. In our saved Nov 13 evidence run (`evidence/bifrost_evidence.json`), Bifrost routed **490** requests and handled **10** automatic failovers across **127,854** tokens."

---

### `1:50 – 2:25` | Agent Lab (PS02) & Real Saved Benchmark Evidence (35s)
- **Exact Clicks:**
  1. Click the **`AGENT LAB`** tab in the top floating navbar.
  2. Show the top KPI strip (**330** recorded runs, **6** architectures, **15** tasks, **100%** `x-bf-run-id` trace coverage from `lab/results/summary.json`, Nov 14, 2026).
  3. Scroll to the **Leaderboard** and highlight `detector_pipeline` (`64.4%` accuracy, `60.0%` consistency) vs `llm_only` (`93.3%` clean, `0.0%` recovery on `model_failure` and `tool_failure`) and `single_search` (`80.0%` clean, `0.0%` recovery on failures).
  4. Click one run in the **Step Trace Inspector** to show individual `search` / `llm` steps, latencies, and token counts.
  5. Briefly switch to **`LEARN` → `Architecture & Evidence`** to show the real saved Bifrost evidence panel (`490` requests, `10` fallbacks, Nov 13, 2026).
- **Narration:**
  > "In **Agent Lab**, we benchmark six architectures—including black-box external agents—on the same 15 tasks and failure modes. Across **330** real runs recorded on Nov 14, unguarded `llm_only` and `single_search` dropped to **0%** recovery under injected tool and model failures, while every run is inspectable step-by-step with `x-bf-run-id` trace correlation."

---

### `2:25 – 2:50` | Freshness Guard (`POST /guard`) & Doc Checker (25s)
- **Exact Clicks:**
  1. Switch to Browser Tab 2 (`http://localhost:8000/docs`, FastAPI OpenAPI UI).
  2. Expand **`POST /guard`**, showing the schema (`question`, `answer`, `context`, `max_claims`), size limits, optional `GUARD_API_KEY`, and per-IP rate limiting.
  3. Briefly show `examples/chatbot_guard_example.py` (~15 lines wrapping any chatbot reply with `from stale_guard import guard`) or switch back to **AUDIT → Doc Checker** to show line-numbered Markdown auditing (`python -m stale_check README.md`).
- **Narration:**
  > "Beyond the UI, any team can wrap their chatbot or RAG pipeline in 15 lines with `stale_guard` or call our rate-limited FastAPI `POST /guard` endpoint—and lint Markdown docs in CI with `python -m stale_check`."

---

### `2:50 – 3:00` | Closing & Links (10s)
- **Screen:** Browser Tab 1 (**AUDIT** tab with completed green/cited result) or Title/Links slide.
- **Exact Clicks:** None — hold clean final frame for 5 seconds.
- **Narration:**
  > "All 350 offline tests pass with zero leaked secrets, and the repo ships ready with Docker, Compose, and offline CI. Check the README for setup and Demo mode—thanks for watching!"

---

## Recording & Post-Production Notes (`TODO`)
- `TODO`: Capture screenshots listed in [`docs/PITCH_OUTLINE.md`](PITCH_OUTLINE.md) into `docs/screenshots/`.
- `TODO`: Upload final 3-minute recording and paste the video URL into `README.md` and `docs/PITCH_OUTLINE.md`.
- `TODO`: Deploy live app and paste the live URL into `README.md` and `docs/PITCH_OUTLINE.md`.
