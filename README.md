# KAIROS
Knowledge-driven Artificial Intelligence for Real-time Operations System 


A personal AI voice assistant. Jarvis-style. Runs on your own hardware.

Listens for a wake word at home, accepts text via Telegram when you're out, accessible from any device via browser. Maintains persistent memory across all sessions. Routes every request to the cheapest model that can handle it.

---

## What it is

- **Voice-first at home** — wake word → speech → response via speaker
- **Text-first outside** — Telegram bot, always on
- **Browser UI** — works from any device on your LAN or via Tailscale
- **Persistent memory** — tasks, events, habits, spending, conversation history
- **Three-tier model routing** — free local model for simple things, cloud only when needed
- **Proactive** — daily briefings, reminders, scheduled tasks without being asked

## What it is not

- Not a multi-user product — built for one person
- Not a chatbot — a persistent agent that remembers and acts
- Not a browser agent or computer-use system
- Not fine-tuned — uses existing models via API routing

---

## Architecture

Six layers, top to bottom:

```
Input channels     voice · telegram · webui · cron
      ↓
Gateway            authenticate · resolve session · normalise to KairosEvent
      ↓
Orchestrator       classify request → pick tier → assemble context → stream response
      ↓
Memory             SQLite (tasks/events/habits/spending) · sqlite-vec (semantic) · session JSON
      ↓
Tools              web search · calendar · messaging · file read
      ↓
Output             TTS → speaker · Telegram sendMessage · WebSocket → browser
```

Every request becomes a `KairosEvent` before anything else touches it. The orchestrator never knows which channel sent the message — only the content and session.

### Three-tier model cascade

Every request is classified by a local model (free, ~100ms) before any cloud call:

| Tier | Model | Latency | Use for |
|------|-------|---------|---------|
| 1 — Local | phi-3-mini via Ollama | ~50ms | Classifier, formatting, trivial Q&A, write-back |
| 2 — Fast cloud | claude-haiku-4-5 | ~300ms | Most voice replies, calendar, web search synthesis |
| 3 — Full power | claude-sonnet-4-6 | ~1s | Complex reasoning, code, research, multi-step planning |

All LLM calls go through LiteLLM proxy. Models are swapped in `litellm/config.yaml` — no code changes needed.

### Memory stack

| Store | Technology | Holds |
|-------|------------|-------|
| Always-on context | `profile.md` + `preferences.json` | Name, tone, goals — prepended to every prompt |
| Structured data | SQLite (`kairos.db`) | Tasks, events, habits, spending — SQL queries |
| Semantic recall | sqlite-vec (inside `kairos.db`) | Conversation history — cosine similarity search |
| Session history | JSON files (`data/sessions/`) | Last N turns per conversation |

---

## Project structure

```
kairos/
├── docker-compose.yml
├── .env                        # API keys — never commit
├── requirements.txt
│
├── runtime/
│   ├── main.py                 # entry point
│   │
│   ├── gateway/
│   │   ├── normalizer.py       # KairosEvent + normalize_* functions
│   │   └── session.py          # session liveness cache
│   │
│   ├── channels/
│   │   ├── voice.py            # Pipecat: VAD → STT → TTS
│   │   ├── telegram.py         # python-telegram-bot adapter
│   │   └── webui.py            # FastAPI + WebSocket
│   │
│   ├── orchestrator/
│   │   ├── orchestrator.py     # main routing logic
│   │   └── classifier.py       # phi-3-mini → intent + tier + domains
│   │
│   ├── memory/
│   │   ├── sqlite_store.py     # tasks, events, habits, spending
│   │   ├── vector_store.py     # sqlite-vec embed + cosine search
│   │   ├── session_store.py    # per-session JSON + compaction
│   │   └── writeback.py        # async fact extract + embed
│   │
│   ├── tools/
│   │   ├── registry.py         # tool schemas + handler mapping
│   │   ├── web_search.py       # Brave API
│   │   ├── calendar.py         # Google Calendar read/write
│   │   └── executor.py         # safe dispatch with allowlist
│   │
│   └── llm/
│       └── client.py           # LiteLLM wrapper, tier-aware, retry
│
├── data/                       # all persistent state
│   ├── profile.md              # your identity context — edit this
│   ├── preferences.json        # sprint, prefs, project states
│   ├── kairos.db               # SQLite: tasks + events + habits + vectors
│   └── sessions/               # one JSON file per session
│
├── static/
│   └── index.html              # browser chat UI
│
└── litellm/
    └── config.yaml             # model routing config
```

---

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) — for local phi-3-mini inference
- [LiteLLM](https://github.com/BerriAI/litellm) — model proxy
- A Telegram bot token (from [@BotFather](https://t.me/BotFather))
- Anthropic API key (for Haiku and Sonnet)

### 1. Clone and install

```bash
git clone <your-repo>
cd kairos
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your keys:

```bash
# LLM providers
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...           # for text-embedding-3-small

# Voice
DEEPGRAM_API_KEY=...
CARTESIA_API_KEY=...
PICOVOICE_ACCESS_KEY=...        # for Porcupine wake word

# Channels
TELEGRAM_BOT_TOKEN=...
TELEGRAM_USER_ID=123456789      # your numeric Telegram ID

# Tools
BRAVE_API_KEY=...
GOOGLE_CALENDAR_CREDENTIALS_PATH=./data/gcal_creds.json

# Internal
LITELLM_BASE_URL=http://localhost:4000
OLLAMA_BASE_URL=http://localhost:11434
DATA_DIR=./data
SESSION_MAX_TURNS=20
```

> **Finding your Telegram user ID:** Message [@userinfobot](https://t.me/userinfobot) on Telegram.

### 3. Edit your profile

`data/profile.md` is the always-on context prepended to every prompt. Keep it under 2KB.

```markdown
# Identity
Name: Your name
Timezone: Asia/Kolkata

# Communication style
- Concise responses unless I ask for detail
- No sycophantic openers
- Casual tone

# Goals this year
- Your goals here

# Do not
- Ask clarifying questions for simple requests
- Repeat what I just said back to me
```

### 4. Start Ollama and pull the local model

```bash
ollama serve
ollama pull phi3               # ~2GB, first time only
```

### 5. Start LiteLLM proxy

```bash
litellm --config litellm/config.yaml --port 4000
```

### 6. Run Kairos

```bash
cd runtime
python main.py
```

You should see:

```
Kairos starting...
Database ready
Orchestrator ready
Kairos is live
  WebUI    → http://localhost:8000
  Telegram → message your bot
Press Ctrl+C to stop
```

---

## Usage

### Browser UI

Open `http://localhost:8000` — type and press Enter. Tokens stream in as they arrive from the LLM.

### Telegram

Message your bot from anywhere. Works over mobile data, in noisy environments, or when you can't speak. Kairos will also send proactive push notifications here (daily briefings, reminders).

### Voice (coming in next step)

Wake word → speak → hear the response. Requires Pipecat, Deepgram, and Cartesia configured.

---

## Design rules

These constraints are non-negotiable — they define what Kairos is:

- **Every response must stream** — never wait for full generation before output begins
- **LLM calls always go through LiteLLM** — never call Anthropic or OpenAI directly
- **Tool inputs validated against schema** before execution — never pass raw LLM output to a tool
- **All I/O is async** — never block the event loop
- **Memory write-back is non-blocking** — `asyncio.create_task()`, not `await`
- **Sessions compact automatically** — never grow unbounded (compacts at 20 turns)
- **Telegram is the only proactive push channel** — voice may not be active when a cron job fires

---

## Latency targets

| Stage | Target | Tool |
|-------|--------|------|
| STT transcription | ~280ms | Deepgram streaming |
| Context assembly | <50ms | SQLite parallel fetch |
| Tier 1 response | ~50ms | phi-3-mini via Ollama |
| Tier 2 first token | ~300ms | claude-haiku-4-5 |
| TTS first chunk | ~80ms | Cartesia |
| **Total (Tier 1)** | **<500ms** | end of speech to first audio |
| **Total (Tier 2)** | **<700ms** | end of speech to first audio |

---

## Remote access

Use [Tailscale](https://tailscale.com) — free for personal use, no port forwarding, works from any device. Install on your laptop and phone, then access the WebUI via your Tailscale IP.

Never expose port 8000 directly to the internet. The WebUI has no authentication layer.

---

## Deployment phases

| Phase | Platform | Notes |
|-------|----------|-------|
| Dev | Laptop | All services local via Docker Compose |
| Always-on | Hetzner CX22 VPS (~€4/month) | 24/7 Telegram polling + cron |
| Self-hosted | Synology / TrueNAS NAS | Full data sovereignty — migrate by rsync of `data/` |

---

## What's not built yet (post-MVP)

- Voice channel (Pipecat + Deepgram + Cartesia)
- Vector memory write-back (sqlite-vec embeddings)
- Tool execution (web search, calendar write, send message)
- Async fact extraction (writeback.py)
- Cron triggers / daily briefings
- Graph memory (Kuzu) — for contacts and project relationships
- WhatsApp channel

---

## Security

- Telegram bot authenticates by numeric user ID — not username (can change)
- Web UI binds to `127.0.0.1` by default — LAN access via Tailscale only
- All API keys in `.env`, never hardcoded, never logged
- Tool inputs validated against JSON schema before execution
- Web content fetched by tools is sanitised before injection into context
- `shell_exec` tool is not in MVP — when added, uses explicit command allowlist only

---

## Tech stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| Async | asyncio throughout |
| Voice pipeline | Pipecat, Deepgram, Cartesia, silero-vad, Porcupine |
| LLM routing | LiteLLM proxy |
| Local inference | Ollama (phi-3-mini) |
| Cloud LLMs | Anthropic (Haiku, Sonnet) |
| Channels | python-telegram-bot, FastAPI + WebSocket |
| Database | SQLite + sqlite-vec |
| Embeddings | text-embedding-3-small (OpenAI) or nomic-embed-text (Ollama) |
| Scheduling | APScheduler |
| Tools | Brave Search API, Google Calendar API |
| Remote access | Tailscale |
| Containers | Docker Compose |