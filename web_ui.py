"""
web_ui.py - Single Page Web Application served at GET / for Stale-Answer Detector.

Renders an interactive, responsive frontend on Vercel with:
- Light and Dark mode switch (persisted in localStorage).
- CSS animations: hover lifts, button glows, radial score ring animation, skeleton pulse.
- 4 pre-loaded Demo examples (iPhone, GPT-4, GST, Company CEO).
- Live verification via POST /verify with automatic demo fallback.
- Side-by-side comparison boxes and explainable source citations.
"""

import json
from demo_data import get_pasted_demo_examples


def get_web_ui_html() -> str:
    """Return the complete standalone HTML document for the Stale-Answer Detector Web UI."""
    raw_demos = get_pasted_demo_examples()
    demos_json = json.dumps(raw_demos, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Stale-Answer Detector</title>
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%230F1524'/%3E%3Cpath d='M32 14v18l12 6' stroke='%236366F1' stroke-width='4.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3Ccircle cx='48' cy='16' r='6' fill='%23F59E0B' stroke='%230F1524' stroke-width='2'/%3E%3C/svg%3E">
  <style>
    :root {{
      --sad-btn-ease: cubic-bezier(0.22, 1, 0.36, 1);
      --sad-btn-duration: 180ms;
      --bg-app: #0B1120;
      --bg-card: #131A2A;
      --bg-subcard: #0E1422;
      --border-card: #243047;
      --border-subcard: #202B40;
      --text-main: #F8FAFC;
      --text-muted: #94A3B8;
      --text-body: #CBD5E1;
      --accent: #6366F1;
      --accent-hover: #7073F7;
      --accent-glow: rgba(99, 102, 241, 0.32);
      --hero-bg: linear-gradient(135deg, #131A2A 0%, #171E35 60%, #1C2042 100%);
      --ring-inner: #0F1524;
      --card-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }}

    [data-theme="light"] {{
      --bg-app: #F8FAFC;
      --bg-card: #FFFFFF;
      --bg-subcard: #F1F5F9;
      --border-card: #E2E8F0;
      --border-subcard: #CBD5E1;
      --text-main: #0F172A;
      --text-muted: #64748B;
      --text-body: #334155;
      --accent: #4F46E5;
      --accent-hover: #4338CA;
      --accent-glow: rgba(79, 70, 229, 0.22);
      --hero-bg: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 60%, #F5F3FF 100%);
      --ring-inner: #FFFFFF;
      --card-shadow: 0 4px 20px rgba(148, 163, 184, 0.18);
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg-app);
      color: var(--text-body);
      line-height: 1.5;
      transition: background-color 0.25s ease, color 0.25s ease;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}

    .container {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 1.5rem 1rem 3rem 1rem;
      width: 100%;
    }}

    /* Top Nav / Theme Bar */
    .top-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.25rem;
    }}
    .brand-link {{
      display: flex;
      align-items: center;
      gap: 0.6rem;
      text-decoration: none;
      color: var(--text-main);
      font-weight: 700;
      font-size: 1.15rem;
    }}
    .brand-icon {{
      width: 32px;
      height: 32px;
      border-radius: 8px;
    }}
    .nav-actions {{
      display: flex;
      align-items: center;
      gap: 0.65rem;
    }}
    .theme-toggle-btn {{
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      background: var(--bg-card);
      border: 1px solid var(--border-card);
      color: var(--text-main);
      padding: 0.42rem 0.85rem;
      border-radius: 999px;
      font-size: 0.84rem;
      font-weight: 600;
      cursor: pointer;
      transition: transform var(--sad-btn-duration) var(--sad-btn-ease),
                  background-color var(--sad-btn-duration) ease,
                  box-shadow var(--sad-btn-duration) ease;
    }}
    .theme-toggle-btn:hover {{
      transform: translateY(-1px);
      box-shadow: 0 4px 12px var(--accent-glow);
    }}
    .theme-toggle-btn:active {{
      transform: translateY(0) scale(0.97);
    }}
    .nav-pill {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      background: var(--bg-card);
      border: 1px solid var(--border-card);
      color: var(--text-muted);
      text-decoration: none;
      padding: 0.42rem 0.85rem;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 600;
      transition: color 0.15s ease, border-color 0.15s ease;
    }}
    .nav-pill:hover {{
      color: var(--accent);
      border-color: var(--accent);
    }}

    /* Hero Banner */
    .sad-hero {{
      background: var(--hero-bg);
      border: 1px solid var(--border-card);
      border-radius: 14px;
      padding: 1.4rem 1.6rem;
      margin-bottom: 1.25rem;
      box-shadow: var(--card-shadow);
      position: relative;
      overflow: hidden;
    }}
    .sad-hero::after {{
      content: "";
      position: absolute;
      top: -50%;
      right: -20%;
      width: 320px;
      height: 320px;
      background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
      pointer-events: none;
    }}
    .sad-hero-title {{
      font-size: 1.85rem;
      font-weight: 800;
      color: var(--text-main);
      letter-spacing: -0.025em;
      margin-bottom: 0.35rem;
    }}
    .sad-hero-tagline {{
      font-size: 0.96rem;
      color: var(--text-muted);
      max-width: 780px;
      margin-bottom: 0.85rem;
    }}
    .sad-hero-badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }}
    .sad-tech-badge {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font-size: 0.76rem;
      font-weight: 600;
      color: var(--accent);
      background: rgba(99, 102, 241, 0.12);
      border: 1px solid rgba(99, 102, 241, 0.35);
      border-radius: 999px;
      padding: 0.22rem 0.68rem;
    }}
    .sad-dot {{
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--accent);
      display: inline-block;
    }}

    /* Quick Demos Section */
    .section-label {{
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 0.5rem;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}
    .demo-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-bottom: 1.15rem;
    }}
    .demo-btn {{
      background: var(--bg-card);
      border: 1px solid var(--border-card);
      color: var(--text-main);
      border-radius: 999px;
      padding: 0.42rem 0.9rem;
      font-size: 0.84rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      transition: transform var(--sad-btn-duration) var(--sad-btn-ease),
                  border-color var(--sad-btn-duration) ease,
                  box-shadow var(--sad-btn-duration) ease,
                  background-color var(--sad-btn-duration) ease;
    }}
    .demo-btn:hover {{
      transform: translateY(-2px);
      border-color: var(--accent);
      box-shadow: 0 4px 14px var(--accent-glow);
    }}
    .demo-btn.active {{
      background: var(--accent);
      color: #FFFFFF;
      border-color: var(--accent);
      box-shadow: 0 4px 14px var(--accent-glow);
    }}

    /* Main Form Card */
    .input-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-card);
      border-radius: 14px;
      padding: 1.25rem 1.4rem;
      margin-bottom: 1.5rem;
      box-shadow: var(--card-shadow);
    }}
    .textarea-wrap {{
      margin-bottom: 0.85rem;
    }}
    textarea.answer-input {{
      width: 100%;
      min-height: 110px;
      background: var(--bg-subcard);
      border: 1px solid var(--border-subcard);
      border-radius: 10px;
      padding: 0.85rem 1rem;
      color: var(--text-main);
      font-family: inherit;
      font-size: 0.94rem;
      line-height: 1.55;
      resize: vertical;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }}
    textarea.answer-input:focus {{
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-glow);
    }}
    .form-options {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
    }}
    .checkbox-group {{
      display: flex;
      align-items: center;
      gap: 1.15rem;
      font-size: 0.84rem;
      color: var(--text-muted);
    }}
    .checkbox-label {{
      display: flex;
      align-items: center;
      gap: 0.4rem;
      cursor: pointer;
      user-select: none;
    }}
    .btn-group {{
      display: flex;
      gap: 0.6rem;
    }}
    .btn-primary {{
      background-color: var(--accent);
      border: 1px solid var(--accent);
      color: #FFFFFF;
      font-weight: 700;
      font-size: 0.9rem;
      padding: 0.6rem 1.4rem;
      border-radius: 9px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      box-shadow: 0 2px 8px var(--accent-glow);
      transition: transform var(--sad-btn-duration) var(--sad-btn-ease),
                  background-color var(--sad-btn-duration) ease,
                  box-shadow 200ms var(--sad-btn-ease);
    }}
    .btn-primary:hover:not(:disabled) {{
      transform: translateY(-2px);
      background-color: var(--accent-hover);
      box-shadow: 0 6px 18px var(--accent-glow);
    }}
    .btn-primary:active:not(:disabled) {{
      transform: translateY(0) scale(0.985);
    }}
    .btn-primary:disabled {{
      opacity: 0.6;
      cursor: not-allowed;
    }}
    .btn-secondary {{
      background: transparent;
      border: 1px solid var(--border-card);
      color: var(--text-muted);
      font-weight: 600;
      font-size: 0.88rem;
      padding: 0.6rem 1rem;
      border-radius: 9px;
      cursor: pointer;
      transition: background-color 0.15s ease, color 0.15s ease;
    }}
    .btn-secondary:hover {{
      background: var(--bg-subcard);
      color: var(--text-main);
    }}

    /* Loading Spinner & Skeleton */
    .spinner {{
      display: inline-block;
      width: 16px;
      height: 16px;
      border: 2px solid rgba(255, 255, 255, 0.35);
      border-radius: 50%;
      border-top-color: #FFFFFF;
      animation: spin 0.8s linear infinite;
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}

    @keyframes sadPulse {{
      0%, 100% {{ opacity: 0.45; }}
      50% {{ opacity: 0.85; }}
    }}
    .skeleton {{
      width: 100%;
      height: 120px;
      border-radius: 12px;
      background: linear-gradient(90deg, var(--bg-card) 25%, var(--bg-subcard) 50%, var(--bg-card) 75%);
      border: 1px solid var(--border-card);
      margin-bottom: 0.9rem;
      animation: sadPulse 1.4s ease-in-out infinite;
    }}

    /* Score Card */
    .sad-score-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-card);
      border-radius: 14px;
      padding: 1.3rem 1.6rem;
      margin-bottom: 1.25rem;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 1.6rem;
      box-shadow: var(--card-shadow);
    }}
    .sad-ring-wrap {{
      flex-shrink: 0;
      position: relative;
      width: 128px;
      height: 128px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .sad-ring-svg {{
      transform: rotate(-90deg);
      width: 128px;
      height: 128px;
    }}
    .sad-ring-bg {{
      fill: none;
      stroke: var(--border-card);
      stroke-width: 10;
    }}
    .sad-ring-fill {{
      fill: none;
      stroke-width: 10;
      stroke-linecap: round;
      stroke-dasharray: 339;
      stroke-dashoffset: 339;
      transition: stroke-dashoffset 1s cubic-bezier(0.22, 1, 0.36, 1), stroke 0.5s ease;
    }}
    .sad-ring-inner {{
      position: absolute;
      width: 96px;
      height: 96px;
      border-radius: 50%;
      background: var(--ring-inner);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
    }}
    .sad-ring-num {{
      font-size: 2rem;
      font-weight: 800;
      color: var(--text-main);
      line-height: 1;
    }}
    .sad-ring-denom {{
      font-size: 0.72rem;
      color: var(--text-muted);
      margin-top: 2px;
    }}
    .sad-ring-tier {{
      font-size: 0.74rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-top: 3px;
    }}
    .sad-score-meta {{
      flex: 1;
      min-width: 250px;
    }}
    .sad-score-title {{
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 0.35rem;
    }}
    .sad-score-stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      margin-top: 0.65rem;
    }}
    .stat-badge {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font-size: 0.78rem;
      font-weight: 600;
      background: var(--bg-subcard);
      border: 1px solid var(--border-subcard);
      padding: 0.22rem 0.65rem;
      border-radius: 6px;
      color: var(--text-main);
    }}

    /* Claim Cards */
    @keyframes fadeInUp {{
      from {{
        opacity: 0;
        transform: translateY(12px);
      }}
      to {{
        opacity: 1;
        transform: translateY(0);
      }}
    }}
    .sad-claim-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-card);
      border-radius: 12px;
      padding: 1.15rem 1.3rem;
      margin-bottom: 0.95rem;
      box-shadow: var(--card-shadow);
      animation: fadeInUp 0.35s cubic-bezier(0.22, 1, 0.36, 1) both;
    }}
    .sad-claim-header {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 0.6rem;
      margin-bottom: 0.75rem;
    }}
    .sad-claim-badges {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.45rem;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      border-radius: 999px;
      padding: 0.18rem 0.62rem;
    }}
    .pill-fresh {{
      color: #22C55E;
      background: rgba(34, 197, 94, 0.14);
      border: 1px solid rgba(34, 197, 94, 0.35);
    }}
    .pill-outdated {{
      color: #F97316;
      background: rgba(249, 115, 22, 0.14);
      border: 1px solid rgba(249, 115, 22, 0.35);
    }}
    .pill-incorrect {{
      color: #EF4444;
      background: rgba(239, 68, 68, 0.14);
      border: 1px solid rgba(239, 68, 68, 0.35);
    }}
    .pill-partially-correct {{
      color: #F59E0B;
      background: rgba(245, 158, 11, 0.14);
      border: 1px solid rgba(245, 158, 11, 0.35);
    }}
    .pill-uncertain {{
      color: #EAB308;
      background: rgba(234, 179, 8, 0.14);
      border: 1px solid rgba(234, 179, 8, 0.35);
    }}
    .pill-unverifiable {{
      color: #94A3B8;
      background: rgba(148, 163, 184, 0.14);
      border: 1px solid rgba(148, 163, 184, 0.35);
    }}

    .risk-pill {{
      font-size: 0.72rem;
      font-weight: 600;
      border-radius: 6px;
      padding: 0.15rem 0.5rem;
    }}
    .risk-critical {{
      color: #F87171;
      background: rgba(239, 68, 68, 0.12);
      border: 1px solid rgba(239, 68, 68, 0.3);
    }}
    .risk-moderate {{
      color: #FBBF24;
      background: rgba(245, 158, 11, 0.12);
      border: 1px solid rgba(245, 158, 11, 0.3);
    }}
    .risk-low {{
      color: var(--text-muted);
      background: rgba(148, 163, 184, 0.12);
      border: 1px solid rgba(148, 163, 184, 0.3);
    }}

    .cat-badge {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font-size: 0.74rem;
      font-weight: 500;
      color: var(--text-body);
      background: var(--bg-subcard);
      border: 1px solid var(--border-subcard);
      border-radius: 6px;
      padding: 0.15rem 0.5rem;
      text-transform: capitalize;
    }}

    .sad-compare-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
      margin-bottom: 0.75rem;
    }}
    @media (max-width: 640px) {{
      .sad-compare-grid {{
        grid-template-columns: 1fr;
      }}
    }}
    .sad-compare-box {{
      background: var(--bg-subcard);
      border: 1px solid var(--border-subcard);
      border-radius: 9px;
      padding: 0.75rem 0.9rem;
    }}
    .sad-compare-actual {{
      border-color: rgba(99, 102, 241, 0.38);
      background: rgba(99, 102, 241, 0.06);
    }}
    .sad-compare-label {{
      font-size: 0.68rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      margin-bottom: 0.3rem;
    }}
    .sad-compare-actual .sad-compare-label {{
      color: var(--accent);
    }}
    .sad-compare-body {{
      font-size: 0.88rem;
      color: var(--text-main);
      line-height: 1.45;
    }}

    .sad-reason {{
      font-size: 0.86rem;
      color: var(--text-body);
      line-height: 1.45;
      margin-bottom: 0.65rem;
    }}
    .sad-chips-row {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.45rem;
    }}
    .sad-source-chip {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font-size: 0.75rem;
      font-weight: 500;
      color: var(--accent);
      text-decoration: none;
      background: rgba(99, 102, 241, 0.12);
      border: 1px solid rgba(99, 102, 241, 0.3);
      border-radius: 999px;
      padding: 0.2rem 0.65rem;
      transition: transform 180ms var(--sad-btn-ease),
                  background-color 180ms ease,
                  border-color 180ms ease,
                  box-shadow 180ms ease;
    }}
    .sad-source-chip:hover {{
      transform: translateY(-1px);
      background: rgba(99, 102, 241, 0.22);
      border-color: var(--accent);
      box-shadow: 0 3px 10px var(--accent-glow);
    }}
    .sad-source-chip:active {{
      transform: translateY(0) scale(0.975);
    }}
    .src-type-tag {{
      font-size: 0.62rem;
      font-weight: 700;
      text-transform: uppercase;
      border-radius: 999px;
      padding: 0.08rem 0.38rem;
      background: rgba(99, 102, 241, 0.25);
    }}

    /* Footer */
    footer {{
      margin-top: auto;
      border-top: 1px solid var(--border-card);
      padding: 1.25rem 1rem;
      text-align: center;
      font-size: 0.82rem;
      color: var(--text-muted);
    }}
    footer a {{
      color: var(--accent);
      text-decoration: none;
    }}
    footer a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <div class="container">
    <!-- Top Nav / Theme Bar -->
    <header class="top-bar">
      <a href="/" class="brand-link">
        <svg class="brand-icon" viewBox="0 0 64 64">
          <rect width="64" height="64" rx="14" fill="#0F1524"/>
          <path d="M32 14v18l12 6" stroke="#6366F1" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>
          <circle cx="48" cy="16" r="6" fill="#F59E0B" stroke="#0F1524" stroke-width="2"/>
        </svg>
        <span>Stale-Answer Detector</span>
      </a>
      <div class="nav-actions">
        <button id="themeToggleBtn" class="theme-toggle-btn" aria-label="Toggle theme">
          <span id="themeIcon">☀️</span>
          <span id="themeLabel">Light mode</span>
        </button>
        <a href="/docs" class="nav-pill" target="_blank">Docs (/docs)</a>
        <a href="/health" class="nav-pill" target="_blank">Health</a>
      </div>
    </header>

    <!-- Hero Banner -->
    <section class="sad-hero">
      <h1 class="sad-hero-title">Audit AI Freshness with Live Sources</h1>
      <p class="sad-hero-tagline">
        Extract time-sensitive claims from LLM answers, verify each claim against live web sources via Tavily and Direct Gemini, and compute explainable freshness scores (0–100).
      </p>
      <div class="sad-hero-badges">
        <span class="sad-tech-badge"><span class="sad-dot"></span> Tavily Search &amp; Extract</span>
        <span class="sad-tech-badge"><span class="sad-dot"></span> Explainable Scoring (0–100)</span>
        <span class="sad-tech-badge"><span class="sad-dot"></span> Vercel Serverless Function</span>
        <span class="sad-tech-badge"><span class="sad-dot"></span> Direct Gemini / Bifrost Ready</span>
      </div>
    </section>

    <!-- Quick Demos -->
    <div class="section-label">⚡ Try a Demo Example</div>
    <div class="demo-bar" id="demoButtons">
      <!-- Demo buttons injected by JS -->
    </div>

    <!-- Main Audit Form -->
    <section class="input-card">
      <div class="textarea-wrap">
        <textarea id="answerInput" class="answer-input" placeholder="Paste an LLM-generated answer text to audit claim-by-claim..."></textarea>
      </div>
      <div class="form-options">
        <div class="checkbox-group">
          <label class="checkbox-label">
            <input type="checkbox" id="useCache" checked> Use Cache
          </label>
          <label class="checkbox-label">
            <input type="checkbox" id="dualJudge"> Dual Judge
          </label>
        </div>
        <div class="btn-group">
          <button id="clearBtn" class="btn-secondary">Clear</button>
          <button id="verifyBtn" class="btn-primary">
            <span id="btnSpinner" class="spinner" style="display: none;"></span>
            <span id="btnText">Check Freshness</span>
          </button>
        </div>
      </div>
    </section>

    <!-- Results Section -->
    <div id="resultsContainer">
      <!-- Populated dynamically -->
    </div>
  </div>

  <footer>
    <p>
      Stale-Answer Detector &bull; Deployed on Vercel &bull; 
      <a href="/docs">OpenAPI Docs</a> &bull; 
      <a href="/health">Health Check</a> &bull; 
      <a href="https://github.com/PurabTomar/stale-detector" target="_blank" rel="noopener">GitHub</a>
    </p>
  </footer>

  <script>
    // Embedded pre-verified offline demos
    const DEMOS = {demos_json};

    // DOM Elements
    const themeToggleBtn = document.getElementById("themeToggleBtn");
    const themeIcon = document.getElementById("themeIcon");
    const themeLabel = document.getElementById("themeLabel");
    const demoButtons = document.getElementById("demoButtons");
    const answerInput = document.getElementById("answerInput");
    const useCache = document.getElementById("useCache");
    const dualJudge = document.getElementById("dualJudge");
    const verifyBtn = document.getElementById("verifyBtn");
    const btnSpinner = document.getElementById("btnSpinner");
    const btnText = document.getElementById("btnText");
    const clearBtn = document.getElementById("clearBtn");
    const resultsContainer = document.getElementById("resultsContainer");

    // Theme Management
    function initTheme() {{
      const saved = localStorage.getItem("theme") || "dark";
      setTheme(saved);
    }}

    function setTheme(mode) {{
      document.documentElement.setAttribute("data-theme", mode);
      localStorage.setItem("theme", mode);
      if (mode === "light") {{
        themeIcon.textContent = "🌙";
        themeLabel.textContent = "Dark mode";
      }} else {{
        themeIcon.textContent = "☀️";
        themeLabel.textContent = "Light mode";
      }}
    }}

    themeToggleBtn.addEventListener("click", () => {{
      const current = document.documentElement.getAttribute("data-theme") || "dark";
      setTheme(current === "dark" ? "light" : "dark");
    }});

    // Initialize Demo Buttons
    function renderDemoButtons() {{
      DEMOS.forEach((demo, idx) => {{
        const btn = document.createElement("button");
        btn.className = "demo-btn" + (idx === 0 ? " active" : "");
        btn.textContent = demo.label;
        btn.addEventListener("click", () => {{
          document.querySelectorAll(".demo-btn").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          loadDemo(demo);
        }});
        demoButtons.appendChild(btn);
      }});

      if (DEMOS.length > 0) {{
        loadDemo(DEMOS[0]);
      }}
    }}

    function loadDemo(demo) {{
      answerInput.value = demo.answer_text;
      renderAuditResult(formatDemoPayload(demo));
    }}

    function formatDemoPayload(demo) {{
      const claims = (demo.results || []).map((r, i) => ({{
        index: i + 1,
        claim: r.claim,
        category: r.category || "general",
        status: r.status || "outdated",
        score: r.score,
        confidence: r.confidence || "high",
        risk_level: r.risk_level || "critical",
        current_info: r.current_info || "",
        current_value: r.current_value || null,
        correction: r.correction || null,
        was_ever_true: r.was_ever_true || null,
        change_date: r.change_date || null,
        effective_date: r.effective_date || null,
        change_summary: r.change_summary || null,
        stale_since: r.stale_since || null,
        stale_duration_label: r.stale_duration_label || null,
        reason: r.reason || "",
        sources: r.sources || [],
        model: r.model || "gemini-2.5-flash",
      }}));

      let totalScore = 0;
      let scoredCount = 0;
      let fresh = 0, outdated = 0, incorrect = 0, partially_correct = 0, uncertain = 0, unverifiable = 0;
      claims.forEach(c => {{
        if (c.status === "fresh") fresh++;
        else if (c.status === "outdated") outdated++;
        else if (c.status === "incorrect") incorrect++;
        else if (c.status === "partially_correct") partially_correct++;
        else if (c.status === "uncertain") uncertain++;
        else if (c.status === "unverifiable") unverifiable++;

        if (typeof c.score === "number") {{
          totalScore += c.score;
          scoredCount++;
        }}
      }});

      const overall = scoredCount > 0 ? Math.round(totalScore / scoredCount) : 0;
      let tier = "Fresh";
      if (overall < 50) tier = "Stale";
      else if (overall < 80) tier = "Mixed";

      return {{
        app: "Stale-Answer Detector",
        routing_mode: "direct_gemini (demo)",
        summary: {{
          overall_score: overall,
          overall_tier: tier,
          claims_checked: claims.length,
          fresh: fresh,
          outdated: outdated,
          incorrect: incorrect,
          partially_correct: partially_correct,
          uncertain: uncertain,
          unverifiable: unverifiable,
          tavily_credits_used: 0.0,
        }},
        tokens_used: {{
          total_tokens: 0,
          requests_count: claims.length,
          avg_latency_ms: 0.0,
          audit_duration_ms: 0.0
        }},
        claims: claims
      }};
    }}

    // Render Full Audit Result Cards
    function renderAuditResult(data) {{
      const summary = data.summary || {{}};
      const score = typeof summary.overall_score === "number" ? summary.overall_score : null;
      const tier = summary.overall_tier || "Unverifiable";
      const claims = data.claims || [];

      // Determine ring color
      let strokeColor = "#22C55E";
      let tierColor = "#22C55E";
      if (score === null || tier === "Unverifiable") {{
        strokeColor = "#64748B";
        tierColor = "#64748B";
      }} else if (score < 50) {{
        strokeColor = "#EF4444";
        tierColor = "#EF4444";
      }} else if (score < 80) {{
        strokeColor = "#F59E0B";
        tierColor = "#F59E0B";
      }}

      // Calculate stroke-dashoffset: circumference = 2 * PI * 54 = 339.29
      const maxOffset = 339.29;
      const offset = score !== null ? maxOffset - (maxOffset * score / 100) : maxOffset;

      let verdictParts = [];
      if (summary.incorrect > 0) verdictParts.push(`${{summary.incorrect}} incorrect claim${{summary.incorrect === 1 ? '' : 's'}}`);
      if (summary.outdated > 0) verdictParts.push(`${{summary.outdated}} outdated claim${{summary.outdated === 1 ? '' : 's'}}`);
      if (summary.partially_correct > 0) verdictParts.push(`${{summary.partially_correct}} partially correct claim${{summary.partially_correct === 1 ? '' : 's'}}`);
      let verdictSentence = "";
      if (verdictParts.length > 0) {{
        verdictSentence = verdictParts.join(" and ") + " found.";
      }} else if ((summary.fresh || 0) === (summary.claims_checked || 0) && (summary.claims_checked || 0) > 0) {{
        verdictSentence = `All ${{summary.claims_checked}} claims verified fresh against live sources.`;
      }} else {{
        verdictSentence = `Audited ${{summary.claims_checked || 0}} claim${{summary.claims_checked === 1 ? '' : 's'}} with verified live sources.`;
      }}

      let html = `
        <div class="sad-score-card">
          <div class="sad-ring-wrap">
            <svg class="sad-ring-svg" viewBox="0 0 128 128">
              <circle class="sad-ring-bg" cx="64" cy="64" r="54"></circle>
              <circle id="scoreRingCircle" class="sad-ring-fill" cx="64" cy="64" r="54" style="stroke: ${{strokeColor}}; stroke-dashoffset: ${{offset}};"></circle>
            </svg>
            <div class="sad-ring-inner">
              <span class="sad-ring-num">${{score !== null ? score : "&ndash;"}}</span>
              <span class="sad-ring-denom">/100</span>
              <span class="sad-ring-tier" style="color: ${{tierColor}};">${{tier}}</span>
            </div>
          </div>
          <div class="sad-score-meta">
            <h2 class="sad-score-title">Freshness Audit Report</h2>
            <p style="color: var(--text-main); font-weight: 600; font-size: 0.95rem; margin-bottom: 0.4rem;">
              ${{verdictSentence}}
            </p>
            <div class="sad-score-stats">
              <span class="stat-badge"><span class="pill pill-fresh" style="padding: 0 0.35rem;">Fresh</span> ${{summary.fresh || 0}}</span>
              <span class="stat-badge"><span class="pill pill-outdated" style="padding: 0 0.35rem;">Outdated</span> ${{summary.outdated || 0}}</span>
              <span class="stat-badge"><span class="pill pill-incorrect" style="padding: 0 0.35rem;">Incorrect</span> ${{summary.incorrect || 0}}</span>
              <span class="stat-badge"><span class="pill pill-partially-correct" style="padding: 0 0.35rem;">Partially</span> ${{summary.partially_correct || 0}}</span>
              <span class="stat-badge"><span class="pill pill-uncertain" style="padding: 0 0.35rem;">Uncertain</span> ${{summary.uncertain || 0}}</span>
              <span class="stat-badge"><span class="pill pill-unverifiable" style="padding: 0 0.35rem;">Unverifiable</span> ${{summary.unverifiable || 0}}</span>
            </div>
          </div>
        </div>
      `;

      if (claims.length === 0) {{
        html += `
          <div class="sad-claim-card" style="text-align: center; padding: 2rem;">
            <p style="color: var(--text-muted);">No time-sensitive claims were extracted from this answer text.</p>
          </div>
        `;
      }} else {{
        claims.forEach((c, idx) => {{
          const statusClass = c.status === "fresh" ? "pill-fresh" : (c.status === "outdated" ? "pill-outdated" : (c.status === "incorrect" ? "pill-incorrect" : (c.status === "partially_correct" ? "pill-partially-correct" : (c.status === "uncertain" ? "pill-uncertain" : "pill-unverifiable"))));
          const riskClass = c.risk_level === "critical" ? "risk-critical" : (c.risk_level === "moderate" ? "risk-moderate" : "risk-low");
          const scoreLabel = typeof c.score === "number" ? c.score + "/100" : "Unverified";

          let sourcesHtml = "";
          if (c.sources && c.sources.length > 0) {{
            sourcesHtml = c.sources.map(s => `
              <a href="${{s.url}}" target="_blank" rel="noopener" class="sad-source-chip">
                <span class="src-type-tag">${{s.source_type || 'Source'}}</span>
                <span>${{escapeHtml(s.title || s.url)}}</span>
              </a>
            `).join("");
          }} else {{
            sourcesHtml = `<span style="font-size: 0.76rem; color: var(--text-muted);">No external URLs cited</span>`;
          }}

          let actualMetaHtml = "";
          if (c.status === "incorrect") {{
            const corrVal = c.correction || c.current_value || c.current_info || "Contradicted by authoritative sources";
            let topSrcHtml = "";
            if (c.sources && c.sources.length > 0 && c.sources[0].url) {{
              const topSrcTitle = c.sources[0].title || c.sources[0].url;
              topSrcHtml = ` (<a href="${{c.sources[0].url}}" target="_blank" rel="noopener" style="color: #EF4444; text-decoration: underline;">${{escapeHtml(topSrcTitle)}}</a>)`;
            }}
            actualMetaHtml = `<div style="margin-top: 0.35rem; font-size: 0.8rem; font-weight: 700; color: #EF4444;">Never true / contradicted by: <span style="font-weight: 500; color: var(--text-main);">${{escapeHtml(corrVal)}}</span>${{topSrcHtml}}</div>`;
          }} else {{
            if (c.current_value) {{
              actualMetaHtml += `<div style="margin-top: 0.35rem; font-size: 0.78rem; font-weight: 600; color: var(--accent);">Current Value: ${{escapeHtml(c.current_value)}}</div>`;
            }}
            if (c.status === "outdated" && c.stale_duration_label) {{
              actualMetaHtml += `<div style="font-size: 0.74rem; color: var(--text-muted);">${{escapeHtml(c.stale_duration_label)}}</div>`;
            }}
          }}

          html += `
            <div class="sad-claim-card" style="animation-delay: ${{idx * 0.08}}s;">
              <div class="sad-claim-header">
                <div class="sad-claim-badges">
                  <span class="cat-badge">📂 ${{c.category}}</span>
                  <span class="pill ${{statusClass}}">${{c.status}}</span>
                  <span class="risk-pill ${{riskClass}}">Risk: ${{c.risk_level}}</span>
                </div>
                <div style="font-size: 0.84rem; font-weight: 700; color: var(--text-main);">
                  Score: ${{scoreLabel}}
                </div>
              </div>

              <div class="sad-compare-grid">
                <div class="sad-compare-box">
                  <div class="sad-compare-label">Claimed in Answer</div>
                  <div class="sad-compare-body">${{escapeHtml(c.claim)}}</div>
                </div>
                <div class="sad-compare-box sad-compare-actual">
                  <div class="sad-compare-label">Actually (Today)</div>
                  <div class="sad-compare-body">
                    ${{escapeHtml(c.current_info || 'No conflicting updates found.')}}
                    ${{actualMetaHtml}}
                  </div>
                </div>
              </div>

              ${{c.reason ? `<p class="sad-reason"><strong>Reason:</strong> ${{escapeHtml(c.reason)}}</p>` : ''}}

              <div class="sad-chips-row">
                <span style="font-size: 0.74rem; font-weight: 600; color: var(--text-muted);">Sources:</span>
                ${{sourcesHtml}}
              </div>
            </div>
          `;
        }});
      }}

      resultsContainer.innerHTML = html;
    }}

    function escapeHtml(str) {{
      return String(str || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }}

    // Check Freshness Action (Calls POST /verify with demo fallback)
    verifyBtn.addEventListener("click", async () => {{
      const text = answerInput.value.trim();
      if (!text) {{
        alert("Please paste or type an answer to check.");
        return;
      }}

      // Check if text matches one of the loaded demos exactly
      const matchedDemo = DEMOS.find(d => d.answer_text.trim() === text);
      if (matchedDemo) {{
        renderAuditResult(formatDemoPayload(matchedDemo));
        return;
      }}

      // Show loading state
      btnSpinner.style.display = "inline-block";
      btnText.textContent = "Auditing claims...";
      verifyBtn.disabled = true;
      resultsContainer.innerHTML = `
        <div class="skeleton"></div>
        <div class="skeleton"></div>
      `;

      try {{
        const resp = await fetch("/verify", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{
            answer: text,
            use_cache: useCache.checked,
            dual_judge: dualJudge.checked,
          }})
        }});

        if (!resp.ok) {{
          const errData = await resp.json().catch(() => ({{ detail: "Failed to audit answer." }}));
          // If server returns API key error, fall back gracefully to closest demo with helpful banner
          if (resp.status >= 500 && (errData.detail || "").toLowerCase().includes("api key")) {{
            const fallback = DEMOS[0];
            const payload = formatDemoPayload(fallback);
            renderAuditResult(payload);
            const notice = document.createElement("div");
            notice.style = "background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 1rem; color: #F59E0B; font-size: 0.85rem;";
            notice.innerHTML = "<strong>Notice:</strong> " + escapeHtml(errData.detail) + " Showing demo sample audit.";
            resultsContainer.insertBefore(notice, resultsContainer.firstChild);
            return;
          }}
          throw new Error(errData.detail || "Error " + resp.status);
        }}

        const data = await resp.json();
        renderAuditResult(data);
      }} catch (err) {{
        resultsContainer.innerHTML = `
          <div class="sad-claim-card" style="border-color: #EF4444; background: rgba(239, 68, 68, 0.08);">
            <h3 style="color: #EF4444; margin-bottom: 0.4rem;">Verification Failed</h3>
            <p style="color: var(--text-body); font-size: 0.88rem;">${{escapeHtml(err.message)}}</p>
          </div>
        `;
      }} finally {{
        btnSpinner.style.display = "none";
        btnText.textContent = "Check Freshness";
        verifyBtn.disabled = false;
      }}
    }});

    clearBtn.addEventListener("click", () => {{
      answerInput.value = "";
      resultsContainer.innerHTML = "";
      document.querySelectorAll(".demo-btn").forEach(b => b.classList.remove("active"));
    }});

    // Initialize
    initTheme();
    renderDemoButtons();
  </script>
</body>
</html>
"""
