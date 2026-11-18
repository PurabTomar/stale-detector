# Stale-Answer Detector & Agent Lab — Pitch Deck Outline

> **Data Integrity Note:** Every metric in this deck comes directly from committed result files ([`evidence/bifrost_evidence.json`](../evidence/bifrost_evidence.json) recorded **Nov 13, 2026**, and [`lab/results/summary.json`](../lab/results/summary.json) generated **Nov 14, 2026**). Items awaiting a live run or deployment URL are explicitly marked `TODO`.

---

## Slide 1 — Title & Hook
- **Title:** Don't Trust AI Blindly. Check Its Expiry Date.
- **Subtitle:** Stale-Answer Detector (PS06) + Agent Lab (PS02) — Powered by Tavily & Bifrost (Maxim AI)
- **Visual:** App hero / main Audit workspace header with dark glass UI.
- **Speaker Notes:** "LLMs answer confidently even when the world has moved on. Today we present Stale-Answer Detector and Agent Lab: a live fact-freshness verifier, drop-in RAG freshness guard, and framework-independent agent evaluation harness."
- **Screenshot to Capture:** `docs/screenshots/01-hero-audit-workspace.png` — Main **AUDIT** tab at `1440x900` showing the top navigation bar, routing/quota chips, 5-stage stepper, and always-open **Input Studio**.

---

## Slide 2 — The Problem
- **Key Bullets:**
  - LLM training cutoffs make prices, software versions, tax slabs, interest rates, and leadership titles go stale silently.
  - Users cannot distinguish *fresh* facts from *outdated* facts or *never-true* hallucinations inside a fluent answer.
  - Meanwhile, engineering teams evaluating AI agents lack consistent, framework-independent benchmarks that measure repeatability, step traces, and failure recovery.
- **Speaker Notes:** "Ask any chatbot about GST slabs, flagship phone prices, or API token rates—it blends yesterday's truth with today's facts in one confident paragraph. Worse, when teams try to fix this with agent pipelines, they rarely benchmark whether a ReAct loop or multi-step verifier actually beats a simpler setup under real outages."
- **Screenshot to Capture:** `docs/screenshots/02-annotated-stale-highlights.png` — Side-by-side **LLM Answer** panel with orange (`Outdated`), red (`Incorrect`), and green (`Fresh`) inline highlights over a sample answer.

---

## Slide 3 — The Solution
- **Key Bullets:**
  - **Audit Workspace (PS06):** Extracts atomic claims (including Hindi & Hinglish), scores volatility, verifies live against primary sources, and produces explainable `0–100` freshness scores.
  - **Cited Corrections:** Rewrites outdated/incorrect facts with numbered `[1]`, `[2]` primary-source links while leaving unverified claims untouched and marked `[unverified]`.
  - **Freshness Guard (`stale_guard` + `POST /guard`):** Wraps any chatbot or RAG reply in ~15 lines of Python or one HTTP call.
  - **Agent Lab (PS02):** Benchmarks internal and black-box external agents on identical tasks, repeats, and failure scenarios with full step traces.
- **Speaker Notes:** "We solved both problems in one cohesive stack: verify and cite live facts for end users and apps, and rigorously measure how different agent architectures behave on the exact same verification tasks."
- **Screenshot to Capture:** `docs/screenshots/03-cited-corrected-answer.png` — **Corrected answer** side-by-side red/green word diff plus the **Cited corrected answer & primary sources** card showing `[1]`, `[2]` links and an `[unverified]` badge.

---

## Slide 4 — System Architecture
- **Key Bullets:**
  - **Inputs:** Streamlit 5-mode Input Studio, FastAPI (`POST /verify`, `POST /guard`), `stale_guard` Python SDK, `stale_check` Doc CLI, and Agent Lab CLI/UI.
  - **LLM Layer (`llm.py`):** Task-routed via **Bifrost AI Gateway** (`LLM_MODEL_EXTRACT`, `LLM_MODEL_JUDGE`, cross-provider `extra_body.fallbacks`, `x-bf-vk`, `x-bf-run-id`) or **Direct Gemini** cloud fallback.
  - **Evidence Layer (`verifier.py`):** Primary-first **Tavily Search** + best-URL **Tavily Extract** + 3-worker parallel pool + thread-safe 6h SQLite cache.
  - **Trust & Hardening:** Prompt-injection data tags, strict JSON schema validation, HTML/URL XSS escaping, SSRF guards, and honest `verification_failed` exclusion.
- **Speaker Notes:** "Every stage is cleanly decoupled. Extraction runs on a lightweight model, Tavily fetches official primary domains first with a 6-hour cache, and judging runs single or dual-model through Bifrost—with clean failure states if any upstream API goes down."
- **Screenshot to Capture:** `docs/screenshots/04-architecture-and-evidence.png` — **LEARN → Architecture & Evidence** sub-tab showing dynamic route-aware pipeline diagram and Bifrost recorded evidence metrics.

---

## Slide 5 — Live Product Demo Highlights
- **Key Bullets:**
  - Real-event **"Check Freshness"** animation: scanning beam → popping claim chips (amber flagged / grey stable) → live `Search → Extract → Judge` substeps with real `Sources read: N` → SVG draw-on score ring & stale highlight wipe.
  - **Stop button** cancels in-flight verification cleanly and shows partial results.
  - **Honest failure UX:** Failed claims show `Verification failed`, score `—`, red stepper step (later steps pending), and **Retry this claim**.
  - **One-click exports:** Markdown, JSON, and offline self-contained HTML with clickable `[1]`, `[2]` citations.
- **Speaker Notes:** "Notice there are zero fake progress bars—every shimmer, substep pill, and source count reflects a real Tavily or LLM callback, and users can stop mid-audit without losing completed claims."
- **Screenshot to Capture:** `docs/screenshots/05-live-verify-and-results.png` — Composite or 2-shot capture of (a) live verify skeleton cards with `Sources read: N` + **Stop** button, and (b) completed **Verification Summary** score ring + claim cards.

---

## Slide 6 — How Tavily Contributes
- **Key Bullets:**
  - **Primary-first entity routing:** Prioritizes official domains (`apple.com`, `openai.com`, `python.org`, `gstcouncil.gov.in`, `rbi.org.in`, `.gov` portals) before general web results.
  - **Office-holder protection:** Leadership claims verify strictly against official government/company pages and widely reputable news outlets.
  - **Deep extraction (`tavily.extract()`):** Confirms exact prices, effective dates, and thresholds on the highest-authority page—and powers **Audit a URL**.
  - **Real saved telemetry (`lab/results/runs.jsonl`, Nov 14, 2026):** **435** tool/search steps recorded across **330** Agent Lab runs (`detector_pipeline`: `1.36` mean tool calls/run; `single_search`: `1.0`; `react_loop`: `0.98`), with 6h SQLite caching conserving credits.
  - **PS06 30-claim benchmark Tavily credits:** `TODO` (pending live `benchmark/results.json`).
- **Speaker Notes:** "Tavily isn't just an extra search call—it drives our entire primary-source authority model, domain deduplication, conflict detection, and URL auditing."
- **Screenshot to Capture:** `docs/screenshots/06-claim-card-primary-sources.png` — Expanded claim card showing single-label `PRIMARY` / `NEWS` domain chips (`+N pages`), `N of M independent sources agree`, and **Evidence snippets**.

---

## Slide 7 — How Bifrost (Maxim AI) Contributes
- **Key Bullets:**
  - **Task specialization:** Separate extract and judge models (`LLM_MODEL_EXTRACT` / `LLM_MODEL_JUDGE`) + **Dual-Judge Consensus** over shared evidence.
  - **Cross-provider failover & outage testing:** Native `extra_body.fallbacks`, `x-bf-vk` virtual keys, and sidebar **Simulate provider outage** toggle.
  - **Real recorded evidence (`evidence/bifrost_evidence.json`, Nov 13, 2026, Bifrost v1.4.0):**
    - **490** total gateway requests across **330** runs
    - **10** automatic failovers (`gemini-3.1-flash-lite` → `gemini-3.5-flash`, `failover_success`)
    - **127,854** total tokens routed (`100,710` in / `27,144` out)
    - **8,011.43 ms** mean gateway latency
    - **330 / 330** Agent Lab runs tagged with `x-bf-run-id` (`100%` trace coverage in `lab/results/summary.json`)
- **Speaker Notes:** "Bifrost gave us production-grade routing, virtual keys, and trace correlation across 490 real requests—including 10 real automatic failovers from flash-lite to flash without crashing a single audit."
- **Screenshot to Capture:** `docs/screenshots/07-bifrost-failover-and-logs.png` — Sidebar **Bifrost Gateway** panel showing `Fallback used: A -> B` alongside the local Bifrost dashboard logs (`http://localhost:8080`).

---

## Slide 8 — Benchmark & Failure-Test Results (Real Saved Data)
- **Key Bullets (from `lab/results/summary.json`, generated Nov 14, 2026 — 330 Runs, 15 Tasks, 6 Architectures):**
  - **`detector_pipeline` (clean, `n=45`):** **64.4%** accuracy, **60.0%** consistency, `3.78` mean steps, `623.9` mean tokens, `33.27s` mean latency. On `gemini-3.5-flash` (`n=3` in `same_agent_matrix`): **100.0%** accuracy (`29.6s` avg).
  - **`llm_only` (`n=75`):** **93.3%** on clean (`n=45`), **0.0%** recovery on `tool_failure` (`n=15`) and `model_failure` (`n=15`) → **56.0%** overall.
  - **`single_search` (`n=75`):** **80.0%** on clean (`n=45`), **0.0%** recovery on `tool_failure` (`n=15`) and `model_failure` (`n=15`) → **48.0%** overall.
  - **`react_loop` (clean, `n=45`):** **48.9%** accuracy, **40.0%** consistency, `787.8` mean tokens, `48.54s` mean latency.
  - **`external_http` & `external_agent` (`n=45` each):** Registered black-box adapters (`0.0%` in offline placeholder run; live external server benchmark: `TODO`).
  - **Honest `TODO` gaps:** 30-claim `benchmark/results.json` live run (`TODO`), `misleading_instructions` live Agent Lab run (`TODO`), USD cost pricing table run (`TODO`), and recorded `evidence/showdown_results.json` (`TODO`).
- **Speaker Notes:** "We report only real saved runs. On Nov 14, 330 runs showed why structured verification matters—and why failure testing matters: unguarded `llm_only` and `single_search` dropped to 0% under model and tool failure injection. We also explicitly mark unrun suites as TODO rather than inventing numbers."
- **Screenshot to Capture:** `docs/screenshots/08-agent-lab-leaderboard-and-traces.png` — **AGENT LAB** tab showing the 330-run leaderboard, same-model comparison matrix, and step-by-step trace inspector.

---

## Slide 9 — Impact & Real-World Use Cases
- **Key Bullets:**
  - **Production RAG & Chatbots (`stale_guard` + `POST /guard`):** Catch stale prices, policies, or leadership answers before users see them—in ~15 lines of Python.
  - **Developer Documentation CI (`stale_check`):** Lint `README.md` and API docs in pull requests with exact line numbers (`L42`).
  - **Enterprise & Multilingual Teams:** Supports English, Hindi (Devanagari), and Hinglish inputs with English web grounding, plus Batch Audit (`.txt`/`.md`/`.csv`) for bulk QA.
  - **AI Engineering Teams (`Agent Lab`):** Compare internal and third-party agents transparently before shipping.
- **Speaker Notes:** "This isn't just a demo UI—it ships as a Python SDK, a rate-limited FastAPI guardrail, a docs CLI linter, and an agent benchmark lab."
- **Screenshot to Capture:** `docs/screenshots/09-guard-openapi-and-doc-checker.png` — FastAPI `/docs` showing `POST /guard` schema alongside **AUDIT → Doc Checker** line-numbered stale report.

---

## Slide 10 — Scalability & Deployment Readiness
- **Key Bullets:**
  - **Concurrency & Caching:** 3-worker parallel verification pool + thread-safe 6-hour SQLite cache (reduces duplicate Tavily credit burn to `0`).
  - **API Protection:** Input size limits, optional `GUARD_API_KEY` env auth, and in-memory per-IP rate limiting (`429` + `Retry-After`) on `POST /guard`.
  - **Containers & CI:** `python:3.12-slim` non-root `Dockerfile` with `HEALTHCHECK`, `docker-compose.yml` (`app` + optional `bifrost` & `api` profiles), `Makefile`, and offline GitHub Actions CI (**350 / 350** offline tests passing, `0` secrets).
  - **Honest Verification Note:** Docker/Compose builds and `make` targets were authored for Linux/Docker hosts and not executed locally (no Docker/Make installed on local Windows dev machine).
- **Speaker Notes:** "The codebase is hardened against XSS, prompt injection, SSRF, and leaked secrets, backed by 350 offline mocked unit tests and container-ready packaging."
- **Screenshot to Capture:** `docs/screenshots/10-tests-and-ci-pass.png` — Terminal capture showing `350` offline tests passing and `scripts/check_secrets.py` reporting zero secrets.

---

## Slide 11 — Roadmap
- **Key Bullets:**
  - **Near-term (`TODO` live runs):** Execute & commit live 30-claim `benchmark/results.json`, `evidence/showdown_results.json`, and `misleading_instructions` Agent Lab runs; validate Docker Compose end-to-end on a Docker host.
  - **Medium-term:** Redis-backed distributed rate limiting & shared Tavily cache for horizontally scaled `POST /guard` deployments; USD cost-per-correct-answer Pareto frontier in Agent Lab.
  - **Long-term:** Browser extension, Slack/Teams freshness bot, and scheduled watchlist alerts when previously fresh claims go stale.
- **Speaker Notes:** "Our immediate next milestone is running the remaining live benchmark suites and Docker validation, followed by distributed caching and proactive freshness monitoring."
- **Screenshot to Capture:** `docs/screenshots/11-history-and-watchlist.png` — **HISTORY** tab showing session audit history, separate built-in **Examples** section, and Claim Watchlist.

---

## Slide 12 — Links & Call to Action
- **Key Bullets:**
  - **GitHub Repository:** https://github.com/PurabTomar/stale-detector
  - **Live App:** `TODO: Add deployed app URL`
  - **3-Minute Video Demo:** `TODO: Add demo video URL`
  - **Try Locally in 60s:** `pip install -r requirements.txt && streamlit run app.py` (toggle **Demo mode** for zero-key offline walkthrough).
- **Speaker Notes:** "Thank you! Everything is open-source under the MIT license—try Demo mode locally with zero keys, or drop `stale_guard` into your chatbot in 15 lines."
- **Screenshot to Capture:** `docs/screenshots/12-final-summary-card.png` — Clean dark-glass **Verification Summary** + **Cited corrected answer** view with QR/repo link overlay.
