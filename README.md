# Stale-Answer Detector

A beginner-friendly Streamlit app that checks LLM-generated answers for outdated facts (prices, software/API versions, laws/regulations, company details, and statistics), re-verifies each claim against live web sources using the **Tavily API**, and scores how fresh the answer is.

## Features

- **Automatic Claim Extraction (`extractor.py`)**: Extracts potentially time-sensitive factual claims and categorizes them (`price`, `version`, `regulation`, `company`, `statistic`, `other`) with targeted search queries.
- **Live Web Verification (`verifier.py`)**: Fetches 5 live web results per claim using Tavily, compares them against today's date via the LLM, assigns a `0–100` freshness score (`fresh`, `uncertain`, `outdated`), and attaches the top 3 source links.
- **Bifrost / OpenAI-Compatible Client (`llm.py`)**: Reads `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, and optional `LLM_MODEL_FALLBACKS` from `.env`, retries once per model, automatically falls back on `429`/`500`/`503`/`404`, and reports which model answered.
- **Interactive UI (`app.py`)**: Includes 3 sidebar demo examples, an overall freshness score at the top, color-coded per-claim cards (`green` / `yellow` / `red`), and a **Generate corrected answer** button to rewrite the original text using verified current info.

## Project Structure

- `app.py` — Streamlit web interface
- `extractor.py` — Extracts verifiable, time-sensitive claims from pasted text
- `verifier.py` — Runs Tavily searches, judges claim freshness, and generates corrected answers
- `llm.py` — OpenAI SDK client configured from `.env` with retry-once fallback and JSON parsing helpers
- `requirements.txt` — Python dependencies
- `.env.example` — Sample environment variable template

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure `.env`**

   Copy `.env.example` to `.env` and fill in your API keys:

   ```bash
   cp .env.example .env
   ```

   Example `.env` for OpenAI:

   ```env
   LLM_BASE_URL=https://api.openai.com/v1
   LLM_API_KEY=sk-your-openai-key
   LLM_MODEL=gpt-4o-mini
   LLM_MODEL_FALLBACKS=gpt-4o,gpt-4.1-mini
   TAVILY_API_KEY=tvly-your-tavily-key
   ```

   **Using a Bifrost gateway (no code changes needed):**

   Just update `.env` to point `LLM_BASE_URL` and `LLM_MODEL` at your Bifrost instance:

   ```env
   LLM_BASE_URL=http://localhost:8080/v1
   LLM_API_KEY=bifrost-local
   LLM_MODEL=openai/gpt-4o-mini
   LLM_MODEL_FALLBACKS=anthropic/claude-3-5-sonnet-20241022,gemini/gemini-2.5-flash
   TAVILY_API_KEY=tvly-your-tavily-key
   ```

3. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```
