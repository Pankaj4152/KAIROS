# Architecture

> How Kairos processes every request — from input to response to memory.

**← [Back to README](../README.md)** · [User Guide](GUIDE.md) · [Setup](SETUP.md) · [Contributing](CONTRIBUTING.md) · [Diagrams HTML](architecture.html) · [Diagrams PDF](kairos_architecture.pdf)

---

## Request Pipeline

Every message from every channel follows this exact flow:

```
You say/type something
        ↓
   [ Gateway ]         → Normalize into a KairosEvent (channel-agnostic)
        ↓
   [ Classifier ]      → Local model decides: tier, domains, tools (~100ms)
        ↓
   ┌──────────────────────────────────────────────┐
   │  All three run in PARALLEL (asyncio.gather)  │
   │                                              │
   │  [ Context Assembly ] → fetch relevant data  │
   │  [ Tool Execution ]   → web search, etc.     │
   │  [ Session History ]  → last 8 turns         │
   └──────────────────────────────────────────────┘
        ↓
   [ Prompt Assembly ]  → system + context + history + user message
        ↓
   [ LLM Stream ]       → yield tokens as they arrive
        ↓
   [ Writeback ]         → fire-and-forget: embed + session + facts
        ↓
   Response delivered to your channel
```

### Why this order matters

1. **Classify first** — the classifier runs on a free local model. It tells us *which* data to fetch and *which* model to use. Without this step, we'd either fetch everything (slow) or use the expensive model for everything (costly).

2. **Parallel fetch** — context assembly, tool execution, and history load are independent I/O operations. Running them with `asyncio.gather()` means total time = slowest single fetch, not the sum of all three.

3. **Stream response** — tokens go to the user as they arrive. The user starts reading while the model is still generating.

4. **Writeback last** — memory updates happen *after* the user has their response. It's `asyncio.create_task()`, not `await` — the user never waits for embedding or fact extraction.

---

## Six-Layer Architecture

```
Input channels     voice · telegram · webui · cron
      ↓
Gateway            authenticate · resolve session · normalise to KairosEvent
      ↓
Orchestrator       classify request → pick tier → assemble context → stream response
      ↓
Memory             SQLite (tasks/events/habits/spending) · sqlite-vec (semantic) · session JSON
      ↓
Tools              web search · calendar · messaging
      ↓
Output             TTS → speaker · Telegram sendMessage · WebSocket → browser
```

Every request becomes a `KairosEvent` before anything else touches it. The orchestrator never knows which channel sent the message — only the content and session.

---

## Three-Tier Model Cascade

Every request is classified by a local model (free, ~100ms) before any cloud call. See the [Interactive Diagrams (HTML)](architecture.html) or [PDF](kairos_architecture.pdf) for the full Routing flow.

| Tier | Model | Cost | Latency | Use for |
|------|-------|------|---------|---------|
| 1 — Local | qwen2.5:3b-instruct via Ollama | Free | ~50ms | Classifier, formatting, trivial Q&A, write-back |
| 2 — Local | qwen2.5:7b-instruct via Ollama | Free | ~150ms | Most routing, calendar, web search synthesis |
| 3 — Cloud | gemini-2.5-flash | Low | ~300ms | Complex reasoning, code, research, multi-step planning |

All LLM calls go through [LiteLLM](https://github.com/BerriAI/litellm) proxy. Models are swapped in `litellm/config.yaml` — no code changes needed.

**Failure behaviour:** If the classifier fails (timeout, bad JSON, model error), it defaults to tier 2. The user gets a slightly more expensive response, but never a crash.

---

## Classifier

The classifier runs on **every request** and outputs:

```json
{
  "intent": "question | task | reminder | memory | chitchat | search | code",
  "complexity": 1,
  "domains": ["tasks", "events"],
  "tools_needed": ["web_search"],
  "tier": 2
}
```

| Field | Purpose |
|-------|---------|
| `intent` | What the user wants (for logging and future routing) |
| `complexity` | 1–3 depth of response needed |
| `domains` | Which memory stores to query (only what's relevant) |
| `tools_needed` | Which tools to invoke before the LLM call |
| `tier` | Which model to use |

**Valid domains:** `tasks`, `events`, `habits`, `spending`, `memory`  
**Valid tools:** `web_search`, `calendar_write`, `send_message`

Unknown values from the model are silently dropped — never passed downstream.

---

## Memory System

Four layers, each with a different purpose:

| Layer | Technology | Holds | When used |
|-------|------------|-------|-----------|
| **Always-on context** | `profile.md` + `preferences.json` | Name, tone, goals, habits | Every single prompt |
| **Structured data** | SQLite (`kairos.db`) | Tasks, events, habits, spending | When classifier flags relevant domains |
| **Semantic recall** | sqlite-vec (cosine search) | Conversation history embeddings | When classifier flags `memory` domain |
| **Session history** | JSON files (`data/sessions/`) | Last N turns per conversation | Always — last 8 turns in prompt |

### Structured Data Schema

```sql
tasks    → id, title, due_date, status, project, priority, created_at
events   → id, title, start_time, end_time, location, notes, source
habits   → id, name, last_done, streak, target_frequency
spending → id, amount, category, merchant, date, notes
```

### Session Compaction

Sessions auto-compact at 20 turns (`SESSION_MAX_TURNS` in `.env`). Old turns are summarised and stored; recent turns are kept verbatim. This prevents context from growing unbounded.

### Writeback Pipeline

After every response, a background task runs:

1. **Embed** — vectorise the conversation turn for future semantic search
2. **Session append** — save the turn to the session JSON file
3. **Fact extraction** — extract key facts from the conversation for long-term memory

All three are non-blocking — `asyncio.create_task()`, not `await`.

---

## Tool System

Tools run **before** the LLM generates a response. Their output becomes part of the prompt context.

### Web Search

Pluggable backend design — switch by setting `SEARCH_BACKEND` in `.env`:

| Backend | Free Tier | Key Required |
|---------|-----------|-------------|
| DuckDuckGo | Unlimited | No |
| Brave | 2000/month | Yes |
| Tavily | 1000/month | Yes |
| Serper | 2500 (no expiry) | Yes |

All backends return the same shape (`title`, `body`, `url`) — the formatter produces identical prompt blocks regardless of backend.

### Tool Validation

Tool inputs are validated against a JSON schema before execution. The LLM's raw output never goes directly to a tool — the executor validates and dispatches.

---

## Design Rules

These constraints are non-negotiable — they define what Kairos is:

| Rule | Rationale |
|------|-----------|
| Every response must stream | Never wait for full generation before output begins |
| LLM calls always go through LiteLLM | Never call Anthropic or OpenAI directly |
| Tool inputs validated against schema | Never pass raw LLM output to a tool |
| All I/O is async | Never block the event loop |
| Memory write-back is non-blocking | `asyncio.create_task()`, not `await` |
| Sessions compact automatically | Never grow unbounded |
| Telegram is the only proactive channel | Voice/WebUI may not be active when cron fires |

---

## Latency Targets

| Stage | Target | Tool |
|-------|--------|------|
| STT transcription | ~280ms | Deepgram streaming |
| Context assembly | <50ms | SQLite parallel fetch |
| Tier 1 response | ~50ms | qwen2.5:3b-instruct via Ollama |
| Tier 2 first token | ~150ms | qwen2.5:7b-instruct via Ollama |
| Tier 3 first token | ~300ms | gemini-2.5-flash |
| TTS first chunk | ~80ms | Cartesia |
| **Total (Tier 1)** | **<500ms** | end of speech to first audio |
| **Total (Tier 2)** | **<700ms** | end of speech to first audio |
| **Total (Tier 3)** | **<1200ms** | end of speech to first audio |

---

## Deployment Phases

| Phase | Platform | Notes |
|-------|----------|-------|
| Dev | Laptop | All services local, Docker Compose optional |
| Always-on | Hetzner CX22 VPS (~€4/month) | 24/7 Telegram polling + cron |
| Self-hosted | Synology / TrueNAS NAS | Full data sovereignty — migrate by rsync of `data/` |
