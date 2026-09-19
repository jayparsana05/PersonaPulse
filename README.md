# PersonaPulse 🤖

> **Automated AI Social Media Agent** — Discovers trending tech news, deduplicates with vector memory, drafts platform-tailored posts in your writing style, and publishes with one Telegram tap.

[![LinkedIn Cron](https://github.com/your-username/personapulse/actions/workflows/linkedin_cron.yml/badge.svg)](https://github.com/your-username/personapulse/actions/workflows/linkedin_cron.yml)
[![X Cron](https://github.com/your-username/personapulse/actions/workflows/x_cron.yml/badge.svg)](https://github.com/your-username/personapulse/actions/workflows/x_cron.yml)
[![Keepalive](https://github.com/your-username/personapulse/actions/workflows/keepalive.yml/badge.svg)](https://github.com/your-username/personapulse/actions/workflows/keepalive.yml)

---

## Architecture Overview

```
GitHub Actions (Cron)
       │
       ▼
  src/agent.py  (LangGraph Pipeline)
  ┌──────────────────────────────────────────────────┐
  │  1. Canary Check  ──── LinkedIn token health     │
  │  2. Search        ──── Tavily API                │
  │  3. Deduplicate   ──── Gemini Embeddings +       │
  │                        Supabase pgvector         │
  │  4. Draft         ──── Gemini Flash 2.0          │
  │  5. Store & Alert ──── Supabase + Telegram       │
  └──────────────────────────────────────────────────┘
                                │
                      Telegram Inline Keyboard
                         (Approve / Reject)
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
       Supabase Edge Function           (no action)
       (telegram-webhook/index.ts)
          │          │
          ▼          ▼
       LinkedIn      X (Twitter)
       API v2        API v2
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | LangGraph (Python) |
| LLM / Drafting | Gemini Flash 2.0 (`gemini-2.0-flash`) |
| Embeddings | `gemini-embedding-001` @ 768-dim (L2-normalized) |
| Database | Supabase PostgreSQL + pgvector (HNSW index) |
| News Discovery | Tavily API |
| Image Extraction | BeautifulSoup4 (`og:image`) |
| Approval Channel | Telegram Bot API (Inline Keyboards + Webhooks) |
| LinkedIn Publishing | LinkedIn API v2 (UGC Posts + Image Upload) |
| X Publishing | X API v2 via tweepy (OAuth 1.0a) |
| Serverless Function | Supabase Edge Functions (Deno TypeScript) |
| Automation | GitHub Actions (Cron + `workflow_dispatch`) |

---

## Directory Structure

```
personapulse/
├── .github/
│   └── workflows/
│       ├── linkedin_cron.yml       # Mon/Wed/Fri 09:00 UTC
│       ├── x_cron.yml              # Tue/Thu/Sat 10:00 UTC
│       └── keepalive.yml           # Daily 08:00 UTC (canary + cleanup)
├── supabase/
│   └── functions/
│       └── telegram-webhook/
│           └── index.ts            # Deno Edge Function
├── src/
│   ├── __init__.py
│   ├── config.py                   # Settings dataclass + env validation
│   ├── ingestion.py                # Tavily search + og:image scraping
│   ├── memory.py                   # Gemini embeddings + pgvector dedup
│   ├── llm.py                      # Gemini Flash 2.0 drafting + Telegram alerts
│   ├── publishers.py               # LinkedIn 3-step upload + X tweepy
│   ├── agent.py                    # LangGraph state graph (5 nodes)
│   └── canary.py                   # LinkedIn token health check
├── schema.sql                      # Full DB schema + cleanup function
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup Guide

### Step 1 – Clone & Install

```bash
git clone https://github.com/your-username/personapulse.git
cd personapulse
pip install -r requirements.txt
```

### Step 2 – Configure Environment

```bash
cp .env.example .env
# Edit .env and fill in all API keys
```

See [Environment Variables](#environment-variables) for details.

### Step 3 – Initialize Supabase Database

1. Open your [Supabase SQL Editor](https://supabase.com/dashboard/project/_/sql).
2. Paste and run the full contents of [`schema.sql`](./schema.sql).
3. Also add this helper RPC function (required by `memory.py`):

```sql
CREATE OR REPLACE FUNCTION match_posts_by_embedding(
    query_embedding   VECTOR(768),
    match_threshold   FLOAT,
    match_count       INT
)
RETURNS TABLE (id UUID, cosine_distance FLOAT) AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id,
        (p.embedding <=> query_embedding)::FLOAT AS cosine_distance
    FROM posts p
    WHERE p.status IN ('PUBLISHED', 'PENDING')
      AND (p.embedding <=> query_embedding) < match_threshold
    ORDER BY cosine_distance ASC
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;
```

### Step 4 – Deploy Supabase Edge Function

```bash
# Install Supabase CLI
npm install -g supabase

# Link to your project
supabase login
supabase link --project-ref YOUR_PROJECT_REF

# Deploy the webhook handler
supabase functions deploy telegram-webhook

# Set Edge Function secrets (Layer 1 secret token & Layer 2 chat authorization)
supabase secrets set \
  SUPABASE_SERVICE_ROLE_KEY="your_key" \
  TELEGRAM_BOT_TOKEN="your_token" \
  TELEGRAM_CHAT_ID="your_chat_id" \
  TELEGRAM_WEBHOOK_SECRET="optional_custom_secret" \
  LINKEDIN_ACCESS_TOKEN="your_token" \
  LINKEDIN_AUTHOR_URN="urn:li:person:XXXXX" \
  X_API_KEY="your_key" \
  X_API_SECRET="your_secret" \
  X_ACCESS_TOKEN="your_token" \
  X_ACCESS_SECRET="your_secret"
```

### Step 5 – Register Telegram Webhook (Two-Layer Security)

You can register the webhook with strict secret token validation using either the included Python helper or `curl`:

**Option A: Using the Python setup script (Recommended)**
```bash
# Auto-detects URL and token from .env, derives compliant secret token
python setup_webhook.py

# Check registered webhook status
python setup_webhook.py --info
```

**Option B: Using cURL**
```bash
# Derive secret token (SHA-256 hex digest of bot token satisfies [a-zA-Z0-9_-]{1,256})
SECRET_TOKEN=$(echo -n "<YOUR_BOT_TOKEN>" | shasum -a 256 | awk '{print $1}')

curl -X POST "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setWebhook" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://<PROJECT_REF>.supabase.co/functions/v1/telegram-webhook",
    "secret_token": "'"$SECRET_TOKEN"'",
    "allowed_updates": ["message", "callback_query"]
  }'
```

### Step 6 – Add GitHub Secrets

In your repo: **Settings → Secrets and variables → Actions**, add all variables from `.env.example`.

---

## Running Locally

```bash
# Run with default query
python -m src.agent

# Run with custom query
python -m src.agent "OpenAI GPT-5 announcement"

# Run just the canary check
python -m src.canary
```

---

## LangGraph Pipeline — Node Details

| # | Node | Input → Output |
|---|------|----------------|
| 1 | **Canary Check** | Reads `LINKEDIN_TOKEN_EXPIRY_DATE` → Telegram alert if < 5 days; halts if expired |
| 2 | **Search** | `search_query` → `article` dict (url, title, body, source, published) |
| 3 | **Deduplicate** | `article` → L2-normalized 768-dim `embedding`; exits if cosine similarity > 0.85 |
| 4 | **Draft** | `article` + `style_profile` → `linkedin_draft` + `x_draft` via Gemini Flash 2.0 |
| 5 | **Store & Alert** | Saves to Supabase (PENDING) → Sends Telegram photo+keyboard for approval |

---

## Telegram Approval Flow

1. Pipeline runs → Telegram message with both drafts + image thumbnail.
2. **[✅ Approve & Publish]** → Edge Function fires:
   - `Promise.allSettled()` publishes to LinkedIn + X in parallel.
   - Each `fetch()` wrapped in `AbortSignal.timeout(8000)`.
   - Supabase status → `PUBLISHED` or `PARTIAL_FAILURE`.
   - Telegram message updated with execution report.
3. **[❌ Reject]** → Supabase status → `REJECTED`. Message updated.

---

## Deduplication Logic

| Step | Detail |
|------|--------|
| Model | `gemini-embedding-001` |
| Dimensions | 768 (truncated from full output) |
| Normalization | L2: divide vector by `sqrt(Σ xᵢ²)` |
| Distance metric | Cosine distance (`<=>` in pgvector) |
| Threshold | Similarity > 0.85 = duplicate |
| Index | HNSW (`vector_cosine_ops`) |

---

## Environment Variables

| Variable | Source |
|----------|--------|
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `TAVILY_API_KEY` | [Tavily](https://app.tavily.com/) |
| `SUPABASE_URL` | Supabase Dashboard → Project Settings → API |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase Dashboard → Project Settings → API |
| `TELEGRAM_BOT_TOKEN` | [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | [@userinfobot](https://t.me/userinfobot) |
| `LINKEDIN_ACCESS_TOKEN` | [LinkedIn Developer Portal](https://www.linkedin.com/developers/apps) |
| `LINKEDIN_AUTHOR_URN` | LinkedIn Developer Portal (`urn:li:person:XXXXX`) |
| `LINKEDIN_TOKEN_EXPIRY_DATE` | Your OAuth grant expiry (YYYY-MM-DD) |
| `X_API_KEY` | [X Developer Portal](https://developer.twitter.com/en/portal/dashboard) |
| `X_API_SECRET` | X Developer Portal |
| `X_ACCESS_TOKEN` | X Developer Portal |
| `X_ACCESS_SECRET` | X Developer Portal |

---

## Cron Schedules

| Workflow | Schedule | Posts to |
|----------|----------|----------|
| `linkedin_cron.yml` | Mon/Wed/Fri 09:00 UTC | LinkedIn |
| `x_cron.yml` | Tue/Thu/Sat 10:00 UTC | X (Twitter) |
| `keepalive.yml` | Daily 08:00 UTC | Token canary + DB cleanup |

---

## License

MIT © 2025 PersonaPulse
