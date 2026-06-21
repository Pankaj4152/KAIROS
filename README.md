# KAIROS

**Knowledge-driven Artificial Intelligence for Real-time Operations System**

A personal AI voice assistant. Jarvis-style. Runs on your own hardware.

Listens for a wake word at home, accepts text via Telegram when you're out, accessible from any device via browser. Maintains persistent memory across all sessions. Routes every request through a classifier that decides which tier — and which tools — the request actually needs.

---

## Highlights

- **Voice-first at home** — wake word → speech → response via speaker (Ongoing / Under development)
- **Text-first outside** — Telegram bot, always on
- **Browser UI** — works from any device on your LAN or via Tailscale
- **Persistent memory** — tasks, events, habits, spending, notes, conversation history
- **Three-tier model routing** — local classifier picks tier/domains/tools before any cloud call; tiers map to LiteLLM model aliases configured in `litellm/config.yaml`
- **11 registered tools** — search, calendar, email (read + write), finance, weather, tasks, spending, habits, notes, Telegram push — see the [Tool Guide](docs/TOOL_GUIDE.md)
- **Resilience first** — tier fallback, safe degradation, and guarded tool execution
- **Observability** — Langfuse tracing across classify → context → tool rounds → generation
- **Cloud backup** — optional Cloudflare R2 sync for `kairos.db`, preferences, and session history
- **Proactive** — daily HTML email briefings, reminders, scheduled tasks without being asked

---

## Demo

### WebUI — Chat with Kairos

<p align="center">
  <img src="docs/images/chat_greeting.png" alt="Kairos greeting" width="700">
</p>

### Task Management

<p align="center">
  <img src="docs/images/add_task.png" alt="Adding a task" width="700">
</p>

<p align="center">
  <img src="docs/images/ask_tasks.png" alt="Querying tasks" width="700">
</p>

### Web Search

<p align="center">
  <img src="docs/images/chat_search.png" alt="Web search results" width="700">
</p>

### Telegram Bot

<p align="center">
  <img src="docs/images/telegram_bot_start.png" alt="Telegram bot" width="400">
</p>

### Terminal Startup

<p align="center">
  <img src="docs/images/terminal.png" alt="Kairos startup logs" width="700">
</p>

> See more in the **[User Guide](docs/GUIDE.md)**.

---

## What it is not

- Not a multi-user product — built for one person
- Not a chatbot — a persistent agent that remembers and acts
- Not a browser agent or computer-use system
- Not fine-tuned — uses existing models via API routing

---

## Quick Start

```bash
git clone https://github.com/Pankaj4152/KAIROS.git
cd kairos
pip install -r requirements.txt
cp .env.example .env        # fill in your API keys
```

```bash
litellm --config litellm/config.yaml --port 4000   # terminal 1
cd runtime && python main.py                        # terminal 2
```

> Local Ollama tiers (`ollama serve` + `qwen2.5` models) are optional — see [Tech Stack](#tech-stack) and [docs/SETUP.md](docs/SETUP.md) for current tier routing. The shipped `litellm/config.yaml` routes all three tiers through Gemini by default; point a `tier*` entry at an Ollama model if you want a free local tier.
>
> For full setup instructions, see **[docs/SETUP.md](docs/SETUP.md)**.

---

## Documentation

| Doc | What's inside |
|-----|---------------|
| **[User Guide](docs/GUIDE.md)** | Features, use cases, example prompts — what Kairos can do for you |
| **[Tool Guide](docs/TOOL_GUIDE.md)** | Every registered tool — parameters, actions, examples, failure modes |
| **[Architecture](docs/ARCHITECTURE.md)** | Request pipeline, memory system, model routing, design rules |
| **[Architecture HTML](docs/architecture.html)** | 7 interactive SVG diagrams — open locally or via GitHub Pages |
| **[Architecture PDF](docs/kairos_architecture.pdf)** | Printable/shareable version of all architecture diagrams |
| **[Resilience](docs/RESILIENCE.md)** | Fallback paths, error recovery strategy, and resilience knobs |
| **[Setup Guide](docs/SETUP.md)** | Installation, configuration, environment variables, deployment |
| **[Contributing](docs/CONTRIBUTING.md)** | How to add channels, tools, search backends, and memory domains |
| **[Releases](RELEASES.md)** | Version history and release highlights |

---

## Mark 3

Mark 3 introduces active user automation features: proactive email channel briefings via SMTP (replacing Telegram alerts), full Google Calendar write/update support, Gmail IMAP read + write tools (`check_gmail` / `gmail_actions`), finance/weather tools, agentic tasks/spending/habits/notes tools, Langfuse tracing, optional Cloudflare R2 cloud backup, and concurrency benchmarking enhancements.
See [Release Notes (Mark 3)](docs/RELEASE_NOTES_MARK3.md) and [Version History](RELEASES.md).

---

## Mark 2

Mark 2 introduces resilience-focused behavior: automatic tier fallback, safer tool-loop degradation, and clearer operational diagnostics.
See [Release Notes (Mark 2)](docs/RELEASE_NOTES_MARK2.md) and [Version History](RELEASES.md).

---

## Project Structure

```
kairos/
├── config/
│   ├── settings.py             # centralised app settings
│   └── logging_config.py       # logging setup (console + optional file, noisy-logger quieting)
│
├── runtime/
│   ├── main.py                 # entry point
│   ├── latency_probe.py        # per-tier latency benchmarking CLI
│   ├── gateway/                # request normalisation + sessions
│   ├── channels/                # telegram, webui, voice, email
│   ├── orchestrator/            # classify → build → stream → writeback
│   ├── memory/                   # sqlite, vectors, sessions, writeback
│   ├── tools/                     # search, calendar, email, finance, weather, tasks, spending, habits, notes, messaging
│   ├── llm/                        # LiteLLM client wrapper
│   ├── utils/                       # R2 cloud-backup storage manager
│   └── static/                       # browser chat UI
│
├── data/                       # all persistent state
│   ├── profile.md              # your identity context
│   ├── preferences.json        # learning goals, prefs, daily habits
│   ├── kairos.db                # SQLite: structured + vector + notes storage
│   └── sessions/                # per-session JSON history
│
├── litellm/
│   └── config.yaml             # model routing config
│
├── test/                       # unit & integration tests
│
├── docs/                       # documentation, including docs/tools/ per-tool reference pages
├── .env.example                # environment variable reference
├── requirements.txt
└── docker-compose.yml
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12 |
| Async | asyncio throughout |
| Voice pipeline | Ongoing / Under development (experimental) |
| LLM routing | LiteLLM proxy — `tier1`/`tier2`/`tier3` aliases configurable per model in `litellm/config.yaml` |
| Cloud LLMs | Gemini (`gemini-3.1-flash-lite`, ships as the default for all three tiers) |
| Local inference (optional) | Ollama (qwen2.5:3b-instruct, qwen2.5:7b-instruct) — wire into `litellm/config.yaml` for a free local tier |
| Embeddings | `gemini-embedding-001` via LiteLLM by default; `EMBEDDING_MODEL` env var also accepts OpenAI or `ollama/nomic-embed-text` |
| Channels | python-telegram-bot, FastAPI + WebSocket, SMTP/IMAP (Gmail) |
| Database | SQLite (structured tables + Python-cosine-similarity vector store + notes FTS5) |
| Scheduling | APScheduler |
| Tools | web search (Brave/Tavily/DuckDuckGo/Serper), Google Calendar, Gmail (IMAP read + SMTP/IMAP write), Yahoo Finance, Open-Meteo weather, tasks, spending, habits, notes — see [Tool Guide](docs/TOOL_GUIDE.md) |
| Observability | Langfuse tracing (classify → context assembly → tool rounds → generation) |
| Cloud backup | Cloudflare R2 (`boto3`) — optional, syncs `kairos.db`/preferences/sessions |
| Remote access | Tailscale |
| Containers | Docker Compose |

---

## Security

- Telegram bot authenticates by numeric user ID — not username
- Web UI binds to `127.0.0.1` by default — LAN access via Tailscale only
- All API keys in `.env`, never hardcoded, never logged
- Tool inputs validated against JSON schema before execution
- Web content fetched by tools is sanitised before injection into context
- Gmail tools (`check_gmail`, `gmail_actions`) use a Google App Password, never your account password — `delete_permanent` is irreversible and should always be confirmed with the user first

---

## License

This is a personal project. No license yet.