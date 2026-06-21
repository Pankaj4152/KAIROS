# Setup Guide

> Installation, configuration, and deployment.

**← [Back to README](../README.md)** · [User Guide](GUIDE.md) · [Tool Guide](TOOL_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Contributing](CONTRIBUTING.md) · [Diagrams HTML](architecture.html) · [Diagrams PDF](kairos_architecture.pdf)

---

## Prerequisites

| Requirement | Purpose |
|-------------|---------|
| Python 3.12 | Runtime |
| [LiteLLM](https://github.com/BerriAI/litellm) | Model proxy — all LLM calls go through this |
| Gemini API key | Default model for all three tiers (`tier1`/`tier2`/`tier3` in `litellm/config.yaml`) |
| Telegram bot token | Telegram channel — get from [@BotFather](https://t.me/BotFather) |
| [Ollama](https://ollama.ai) | Optional — only needed if you wire a `qwen2.5` model into a tier in `litellm/config.yaml` for a free local route |

---

## 1. Clone and Install

```bash
git clone https://github.com/Pankaj4152/KAIROS.git
cd kairos
pip install -r requirements.txt
```

---

## 2. Configure Environment

```bash
cp .env.example .env
```

Open `.env` and fill in your API keys. At minimum you need:

| Key | Required for | Where to get it |
|-----|-------------|-----------------|
| `GEMINI_API_KEY` | Gemini models (default for all tiers) | [aistudio.google.dev](https://aistudio.google.dev) |
| `TELEGRAM_BOT_TOKEN` | Telegram channel | [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_USER_ID` | Auth — locks bot to you | [@userinfobot](https://t.me/userinfobot) |

Optional keys (voice, search, calendar, email, cloud backup, tracing) can be added later. See [`.env.example`](../.env.example) for all options.

### Environment Variable Reference

#### LLM Providers
| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Google Gemini models. `litellm/config.yaml` ships two key slots per tier (`GEMINI_API_KEY` / `GEMINI_API_KEY_2`) for basic key rotation under rate limits |
| `OPENAI_API_KEY` | — | Only needed if you set `EMBEDDING_MODEL` to an OpenAI embedding model |

#### Voice (optional)
| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPGRAM_API_KEY` | — | Speech-to-text |
| `CARTESIA_API_KEY` | — | Text-to-speech |
| `PICOVOICE_ACCESS_KEY` | — | Wake word detection (free at picovoice.ai) |

#### Channels
| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | — | From @BotFather |
| `TELEGRAM_USER_ID` | — | Your numeric Telegram ID |
| `SMTP_SERVER` | `smtp.gmail.com` | SMTP server host for Email morning briefings |
| `SMTP_PORT` | `587` | SMTP port (typically 587 for TLS) |
| `SMTP_USERNAME` | — | Username/email address for SMTP credentials |
| `SMTP_PASSWORD` | — | Gmail SMTP App Password |
| `EMAIL_FROM` | — | Sender email address for morning briefings |
| `EMAIL_TO` | — | Destination email address for briefings |

#### Tools & Search
| Variable | Default | Description |
|----------|---------|-------------|
| `SEARCH_BACKEND` | `duckduckgo` | Options: `duckduckgo`, `brave`, `tavily`, `serper` |
| `SEARCH_MAX_RESULTS` | `5` | Max search results per query |
| `BRAVE_API_KEY` | — | Brave Search (optional) |
| `TAVILY_API_KEY` | — | Tavily Search (optional) |
| `SERPER_API_KEY` | — | Serper.dev search (optional) |
| `GOOGLE_CALENDAR_CREDENTIALS_PATH` | `./data/gcal_creds.json` | Google Calendar OAuth credentials.json path |
| `GOOGLE_CALENDAR_ID` | `primary` | Which calendar the `google_calendar` tool operates on by default. Get other IDs from the tool's `list_calendars` action |
| `GMAIL_USER` | — | Gmail address used by both `check_gmail` (IMAP read) and `gmail_actions` (SMTP/IMAP write) |
| `GMAIL_APP_PASSWORD` | — | Google App Password for secure IMAP/SMTP login — not your account password |

> `finance` (Yahoo Finance) and `weather` (Open-Meteo) need no API keys at all.

#### Internal Services
| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_BASE_URL` | `http://localhost:4000` | LiteLLM proxy URL |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama URL — only relevant if a tier in `litellm/config.yaml` points at an Ollama model |

#### App Config
| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `./data` | Where persistent state lives |
| `SESSION_MAX_TURNS` | `20` | Turns before session compaction |
| `EMBEDDING_MODEL` | `ollama/nomic-embed-text` (code default) — the shipped `litellm/config.yaml` instead exposes a `gemini-embedding-001` route as `embedding-model` | Embedding model used for note/conversation semantic search. Set to match whatever embedding route your LiteLLM config defines |

#### Model Mapping
| Variable | Default | Description |
|----------|---------|-------------|
| `TIER1_MODEL` | `tier1` | Must match `litellm/config.yaml` |
| `TIER2_MODEL` | `tier2` | Must match `litellm/config.yaml` |
| `TIER3_MODEL` | `tier3` | Must match `litellm/config.yaml` |

> The shipped `litellm/config.yaml` routes all three tier aliases to `gemini/gemini-3.1-flash-lite` by default — there's no cost/latency difference between tiers out of the box. To get a genuinely free local tier, add a `model_name: tier1` entry pointing at an Ollama model (e.g. `ollama/qwen2.5:3b-instruct`) and run `ollama serve` alongside LiteLLM.

#### Timeouts & Retry
| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_STREAM_TIMEOUT` | `30` | Streaming response timeout (seconds) |
| `LLM_COMPLETE_TIMEOUT` | `15` | Non-streaming response timeout |
| `CLASSIFIER_TIMEOUT` | `10` | Tier-1 classifier timeout (seconds) |
| `LLM_MAX_RETRIES` | `2` | Retry attempts on failure |

#### Resilience
| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_TOOL_ROUNDS` | `5` | Max tool-calling rounds before degrading to plain generation |
<!-- | `TOOL_CALL_MAX_RETRIES` | `2` | Retry attempts for transient tool-call failures |
| `TOOL_CALL_RETRY_BASE_DELAY` | `0.5` | Exponential backoff base delay (seconds) |
| `TOOL_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `3` | Consecutive failures before opening tool circuit |
| `TOOL_CIRCUIT_BREAKER_COOLDOWN_SECONDS` | `60` | Cooldown before retrying an open-circuit tool | -->

#### Scheduling
| Variable | Default | Description |
|----------|---------|-------------|
| `BRIEFING_HOUR` | `7` | Morning briefing hour (24h format) |
| `BRIEFING_MINUTE` | `30` | Morning briefing minute |
| `TIMEZONE` | `Asia/Kolkata` | Your timezone |

#### Observability (Langfuse, optional)
| Variable | Default | Description |
|----------|---------|-------------|
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse project public key. Without it, tracing calls fail silently and Kairos still runs normally |
| `LANGFUSE_SECRET_KEY` | — | Langfuse project secret key |
| `LANGFUSE_HOST` | — | Langfuse instance URL (cloud or self-hosted) |

#### Cloud Backup (Cloudflare R2, optional)
| Variable | Default | Description |
|----------|---------|-------------|
| `S3_ENDPOINT_URL` | — | Cloudflare R2 endpoint URL |
| `S3_ACCESS_KEY_ID` | — | R2 access key ID |
| `S3_SECRET_ACCESS_KEY` | — | R2 secret access key |
| `S3_BUCKET_NAME` | — | R2 bucket name |

> If any of the four R2 variables are missing, `runtime/utils/storage.py` disables cloud sync entirely and logs a warning — `kairos.db`, `preferences.json`, `profile.md`, and session JSON files then live purely on local disk. This is the default state; R2 sync is opt-in.

---

## 3. Edit Your Profile

`data/profile.md` is the always-on context prepended to every prompt. Keep it under 2KB.

```markdown
# Identity
Name: Your name
Timezone: Asia/Kolkata
Role: 3rd-year CS student, AI/ML + systems focus

# Communication style
- Concise and direct. No fluff, no repetition
- Casual tone
- Do not over-explain basics

# Current focus
- Building Kairos (voice agent MVP)
- DSA practice daily
- Andrej Karpathy's Neural Network lecture playlist
- Learn ML hands-on via projects

# Goals (2026)
- Ship Kairos MVP
- Build in public, gain technical credibility
- Consistent DSA + ML fundamentals

# Daily habits
- DSA practice (non-negotiable)
- Maintain diet and gym routine
- Fix and maintain a healthy sleep cycle
- Structured day routine
- Focus on body and mind — not just code

# Do not
- Give generic or vague advice
- Suggest things without an execution path
- Over-explain basics
```

`data/preferences.json` stores structured preferences (learning goals, execution bias, daily habits) plus facts extracted from conversation by the writeback pipeline. See the file for the full schema.

---

## 4. (Optional) Start Ollama for a local tier

Only needed if you've wired an Ollama model into `litellm/config.yaml`. The shipped config routes every tier to Gemini, so this step is skippable for a cloud-only setup.

```bash
ollama serve
ollama pull qwen2.5:3b-instruct
ollama pull qwen2.5:7b-instruct
```

---

## 5. Start LiteLLM Proxy

```bash
litellm --config litellm/config.yaml --port 4000
```

---

## 6. Run Kairos

```bash
cd runtime
python main.py
```

You should see:

```
Kairos starting...
Database ready
Vector store ready
Orchestrator ready
Kairos is live
  WebUI    → http://localhost:8000
  Telegram → message your bot
Press Ctrl+C to stop
```

On startup, Kairos also verifies the email SMTP transport (logs a warning if briefings will fail) and, if R2 env vars are set, syncs down `kairos.db`/`preferences.json`/`profile.md`/session history before any channel accepts traffic.

### Diagnostics

Use the latency probe to compare model routes directly against LiteLLM. You can specify a benchmark prompt scenario (`--benchmark [simple|reasoning|essay|system_design|coding]`) and run concurrent loads (`--concurrency N`):

```bash
python runtime/latency_probe.py --models tier1 tier2 tier3 --benchmark simple --concurrency 1
```

---

## Remote Access

Use [Tailscale](https://tailscale.com) — free for personal use, no port forwarding, works from any device.

1. Install Tailscale on your laptop and phone
2. Access the WebUI via your Tailscale IP

> ⚠️ **Never expose port 8000 directly to the internet.** The WebUI has no authentication layer.

---

## Deployment

| Phase | Platform | Notes |
|-------|----------|-------|
| **Dev** | Laptop | All services local, Docker Compose optional |
| **Always-on** | Hetzner CX22 VPS (~€4/month) | 24/7 Telegram polling + cron |
| **Self-hosted** | Synology / TrueNAS NAS | Full data sovereignty — rsync `data/` to migrate, or enable R2 sync for automatic cloud backup |

### Docker Compose (dev)

```bash
docker-compose up -d
```

This starts LiteLLM and Kairos together (add Ollama to the compose file if you're using a local tier). Edit `docker-compose.yml` for custom port mappings.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `TELEGRAM_BOT_TOKEN not set` | Fill in `.env` — see [step 2](#2-configure-environment) |
| `TELEGRAM_USER_ID not set` | Message [@userinfobot](https://t.me/userinfobot) to get your numeric ID |
| Classifier always returns the default route | Check the classifier's tier-1 model is reachable: `curl http://localhost:4000/health` (and `curl http://localhost:11434/api/tags` if you're routing tier1 to Ollama) |
| LLM calls timing out | Check LiteLLM is running: `curl http://localhost:4000/health` |
| No search results | Default backend is DuckDuckGo (no key needed). Check `SEARCH_BACKEND` in `.env` |
| WebUI not loading | Make sure you're running `python main.py` from the `runtime/` directory |
| Email briefings fail or SMTP errors | Verify `SMTP_USERNAME` and `SMTP_PASSWORD` are correct. Ensure you are using a secure Google **App Password** rather than your master password, and port `587` is open. |
| `check_gmail` or `gmail_actions` cannot connect | Verify `GMAIL_USER` and `GMAIL_APP_PASSWORD` are present and IMAP access is enabled in your Gmail account settings |
| `finance`/`weather` return errors | Both are keyless (Yahoo Finance, Open-Meteo) — check outbound network access, not API keys |
| R2 cloud sync silently not running | Confirm all four `S3_*` variables are set — `storage_manager` disables itself entirely if any are missing, logging a warning at startup |
| No traces showing up in Langfuse | Confirm `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_HOST` are set; tracing fails silently otherwise and Kairos keeps working without it |