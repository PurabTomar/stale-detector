# Minimal Public Bifrost AI Gateway Deployment

This directory contains the production-ready declarative configuration and container definitions for deploying a minimal, secure, public **Bifrost AI Gateway** (by Maxim AI) to host providers like [Render](https://render.com) or [Railway](https://railway.app).

---

## Security & Governance Guardrails (Per `docs.getbifrost.ai`)

1. **Config References (Zero Keys in Git):**
   All upstream provider keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`) and passwords are read dynamically from runtime environment variables via `env.*` references inside `config.json`. No secrets ever touch version control.
2. **Mandatory Virtual Key:**
   The gateway exposes a dedicated virtual key (`vk-stale-detector`) referencing `env.BIFROST_VIRTUAL_KEY`. Strangers cannot invoke LLM models without supplying this key in the `x-bf-vk` or `Authorization: Bearer` header.
3. **Hard Budget Cap:**
   Spending is strictly bounded by a **$5.00 / month** maximum budget (`max_limit: 5.0, reset_duration: "1M"`). If quota is reached, the gateway rejects requests at the boundary before calling upstream providers.
4. **Sliding-Window Rate Limiting:**
   Requests and token consumption are capped at **60 requests/hour** and **100,000 tokens/hour** (`rl-public-gateway`).
5. **Password-Protected Dashboard:**
   The admin dashboard and management API are protected by `governance.auth_config` requiring `BIFROST_ADMIN_USERNAME` and `BIFROST_ADMIN_PASSWORD`. The dashboard is **NOT open to the public**.
6. **Automatic Direct Gemini Fallback:**
   The companion Stale-Answer Detector web application automatically checks gateway health via `GET /health` (a public, whitelisted route). If your Render free tier spins down or the gateway is unreachable, the app seamlessly falls back to Direct Gemini with a prominent banner so users are never interrupted.

---

## Step-by-Step Manual Deployment Guide on Render

Follow these exact manual steps on Render to deploy your gateway:

### Option A: Using Render Blueprints (`render.yaml`) — Recommended

1. **Push Changes to GitHub:**
   Make sure `render.yaml` and the `deploy/bifrost/` folder are committed and pushed to your GitHub repository.
2. **Log in to Render:**
   Navigate to [dashboard.render.com](https://dashboard.render.com).
3. **Create New Blueprint:**
   - Click the **New +** button in the top right and select **Blueprint**.
   - Connect your GitHub repository (`PurabTomar/stale-detector`).
   - Render will detect the root `render.yaml` file automatically.
4. **Fill In Secret Environment Variables:**
   Render will prompt you for the variables marked `sync: false`:
   - `BIFROST_ADMIN_PASSWORD`: Choose a strong password for your Bifrost dashboard.
   - `BIFROST_VIRTUAL_KEY`: Choose an authentication secret for the app (e.g. `sk-bf-stale-detector-prod-2026`).
   - `GEMINI_API_KEY`: Paste your Google Gemini API key from [Google AI Studio](https://aistudio.google.com).
   - `OPENAI_API_KEY` (Optional): Paste your OpenAI API key if you want secondary cross-provider fallbacks.
   - `BIFROST_ENCRYPTION_KEY`: Render will generate a random 32-character encryption key automatically.
5. **Click Apply:**
   Render will build the Docker container and start the web service.
6. **Copy Your Public Gateway URL:**
   Once deployed, Render gives you a public URL (e.g., `https://bifrost-gateway-xxxx.onrender.com`).

---

### Option B: Manual Web Service Setup on Render

1. Go to [dashboard.render.com](https://dashboard.render.com) &rarr; **New +** &rarr; **Web Service**.
2. Select **Build and deploy from a Git repository** and pick your repo.
3. Configure the following fields:
   - **Name:** `bifrost-gateway`
   - **Region:** Any region (e.g. Oregon or Frankfurt)
   - **Branch:** `main` (or your active branch)
   - **Runtime:** `Docker`
   - **Dockerfile Path:** `deploy/bifrost/Dockerfile`
   - **Docker Context:** `deploy/bifrost`
   - **Instance Type:** `Free`
4. Add the following **Environment Variables** in the **Environment** tab:
   | Key | Value | Notes |
   | :--- | :--- | :--- |
   | `APP_HOST` | `0.0.0.0` | Binds to all network interfaces |
   | `APP_PORT` | `10000` | Render default container port |
   | `PORT` | `10000` | Render port routing |
   | `BIFROST_ADMIN_USERNAME` | `admin` | Dashboard login username |
   | `BIFROST_ADMIN_PASSWORD` | `<your-secure-password>` | Dashboard login password |
   | `BIFROST_ENCRYPTION_KEY` | `<random-32-char-string>` | 32-character key for encryption |
   | `BIFROST_VIRTUAL_KEY` | `sk-bf-stale-detector-prod-2026` | Virtual key required for requests |
   | `GEMINI_API_KEY` | `<your-gemini-key>` | Passed to Gemini provider |
   | `OPENAI_API_KEY` | `<your-openai-key>` | Optional secondary provider |
5. In **Advanced Settings**, set **Health Check Path** to `/health`.
6. Click **Create Web Service**.

---

## Verifying Gateway Health

Once Render indicates the service is **Live**, run this in your terminal:

```bash
# Check public health status
curl -i https://<your-render-url>/health

# Expected response: HTTP 200 OK
# {"status":"ok","providers":["gemini","openai"],"version":"..."}
```

---

## Configuring the Stale-Answer Detector App

In your deployed Streamlit Community Cloud app (or local `.env`), add the following settings:

### In Streamlit Community Cloud Secrets (`.streamlit/secrets.toml`):

```toml
# Enable Bifrost routing via your public Render gateway
USE_BIFROST = "true"
BIFROST_URL = "https://<your-render-url>.onrender.com"
LLM_VIRTUAL_KEY = "sk-bf-stale-detector-prod-2026"

# Primary & Fallback API keys
GEMINI_API_KEY = "AIzaSy..."
TAVILY_API_KEY = "tvly-..."
```

### In Local `.env`:

```env
USE_BIFROST=true
BIFROST_URL=https://<your-render-url>.onrender.com
LLM_VIRTUAL_KEY=sk-bf-stale-detector-prod-2026
GEMINI_API_KEY=AIzaSy...
TAVILY_API_KEY=tvly-...
```

---

## Resilience & Automatic Fallback

- The app will automatically connect to `BIFROST_URL` sending `x-bf-vk: <LLM_VIRTUAL_KEY>`.
- If the Render free container is sleeping or temporarily unreachable, the app's built-in health check detects it within `0.8s` and displays a yellow status banner:
  > **⚠️ Bifrost Gateway unreachable at `https://...` — Automatically fallen back to Direct Gemini API.**
- Factual verification audits and Q&A continue running seamlessly without throwing error screens or failing user workflows.
