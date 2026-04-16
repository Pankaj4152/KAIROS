# Setup Guide

> Installation, configuration, and deployment.

**← [Back to README](../README.md)** · [User Guide](GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Contributing](CONTRIBUTING.md) · [Diagrams HTML](architecture.html) · [Diagrams PDF](kairos_architecture.pdf)

---

## Prerequisites

| Requirement | Purpose |
|-------------|---------|
| Python 3.11+ | Runtime |
| [Ollama](https://ollama.ai) | Local qwen2.5 tier models |
| [LiteLLM](https://github.com/BerriAI/litellm) | Model proxy — all LLM calls go through this |
| Telegram bot token | Telegram channel — get from [@BotFather](https://t.me/BotFather) |
| Gemini API key | Tier 3 cloud model (gemini-2.5-flash) |

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
| `GEMINI_API_KEY` | Gemini models | [aistudio.google.dev](https://aistudio.google.dev) |
| `TELEGRAM_BOT_TOKEN` | Telegram channel | [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_USER_ID` | Auth — locks bot to you | [@userinfobot](https://t.me/userinfobot) |

Optional keys (voice, search, calendar) can be added later. See [`.env.example`](../.env.example) for all options.

### Environment Variable Reference

#### LLM Providers
| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Google Gemini models |
| `OPENAI_API_KEY` | — | Only for text-embedding-3-small |

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

#### Tools & Search
| Variable | Default | Description |
|----------|---------|-------------|
| `SEARCH_BACKEND` | `duckduckgo` | Options: `duckduckgo`, `brave`, `tavily`, `serper` |
| `SEARCH_MAX_RESULTS` | `5` | Max search results per query |
| `BRAVE_API_KEY` | — | Brave Search (optional) |
| `TAVILY_API_KEY` | — | Tavily Search (optional) |
| `SERPER_API_KEY` | — | Serper.dev search (optional) |
| `GOOGLE_CALENDAR_CREDENTIALS_PATH` | `./data/gcal_creds.json` | Google Calendar credentials |

#### Internal Services
| Variable | Default | Description |
|----------|---------|-------------|
| `LITELLM_BASE_URL` | `http://localhost:4000` | LiteLLM proxy URL |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama URL |

#### App Config
| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `./data` | Where persistent state lives |
| `SESSION_MAX_TURNS` | `20` | Turns before session compaction |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for vector search |

#### Model Mapping
| Variable | Default | Description |
|----------|---------|-------------|
| `TIER1_MODEL` | `tier1` | Must match `litellm/config.yaml` |
| `TIER2_MODEL` | `tier2` | Must match `litellm/config.yaml` |
| `TIER3_MODEL` | `tier3` | Must match `litellm/config.yaml` |

#### Timeouts & Retry
| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_STREAM_TIMEOUT` | `30` | Streaming response timeout (seconds) |
| `LLM_COMPLETE_TIMEOUT` | `15` | Non-streaming response timeout |
| `CLASSIFIER_TIMEOUT` | `10` | Tier-1 classifier timeout (seconds) |
| `LLM_MAX_RETRIES` | `2` | Retry attempts on failure |

#### Scheduling
| Variable | Default | Description |
|----------|---------|-------------|
| `BRIEFING_HOUR` | `7` | Morning briefing hour (24h format) |
| `BRIEFING_MINUTE` | `30` | Morning briefing minute |
| `TIMEZONE` | `Asia/Kolkata` | Your timezone |

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

`data/preferences.json` stores structured preferences (learning goals, execution bias, daily habits). See the file for the full schema.

---

## 4. Start Ollama

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

### Diagnostics

Use the latency probe to compare model routes directly against LiteLLM:

```bash
python runtime/latency_probe.py --models tier1 tier2 tier3
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
| **Self-hosted** | Synology / TrueNAS NAS | Full data sovereignty — rsync `data/` to migrate |

### Docker Compose (dev)

```bash
docker-compose up -d
```

This starts Ollama, LiteLLM, and Kairos together. Edit `docker-compose.yml` for custom port mappings.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `TELEGRAM_BOT_TOKEN not set` | Fill in `.env` — see [step 2](#2-configure-environment) |
| `TELEGRAM_USER_ID not set` | Message [@userinfobot](https://t.me/userinfobot) to get your numeric ID |
| Classifier always returns tier 2 | Check Ollama is running: `curl http://localhost:11434/api/tags` |
| LLM calls timing out | Check LiteLLM is running: `curl http://localhost:4000/health` |
| No search results | Default backend is DuckDuckGo (no key needed). Check `SEARCH_BACKEND` in `.env` |
| WebUI not loading | Make sure you're running `python main.py` from the `runtime/` directory |
