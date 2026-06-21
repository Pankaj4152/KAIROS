# Architecture

> How Kairos processes every request — from input to response to memory.

**← [Back to README](../README.md)** · [User Guide](GUIDE.md) · [Tool Guide](TOOL_GUIDE.md) · [Setup](SETUP.md) · [Contributing](CONTRIBUTING.md) · [Diagrams HTML](architecture.html) · [Diagrams PDF](kairos_architecture.pdf)

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
   │  [ Pre-LLM Tools ]    → web search           │
   │  [ Session History ]  → last 8 turns         │
   └──────────────────────────────────────────────┘
        ↓
   [ Prompt Assembly ]  → system + context + history + user message
        ↓
   [ Agentic Tool Loop ] → LLM may call any registered tool, results fed back
        ↓
   [ LLM Stream ]       → yield tokens as they arrive
        ↓
   [ Writeback ]         → fire-and-forget: embed + session + facts + cloud sync
        ↓
   Response delivered to your channel
```

Every step is also wrapped in a Langfuse trace (`kairos-process` as the root span, with nested `classifier`, `context-assembly`, and per-round generation spans) — see [Observability](#observability-langfuse) below.

### Why this order matters

1. **Classify first** — the classifier runs on a configurable tier-1 model. It tells us *which* data to fetch and *which* tools/model tier to use. Without this step, we'd either fetch everything (slow) or use the most expensive model for everything (costly).

2. **Parallel fetch** — context assembly, pre-LLM tool execution (currently just `web_search`), and history load are independent I/O operations. Running them with `asyncio.gather()` means total time = slowest single fetch, not the sum of all three.

3. **Agentic tools after the first message is assembled** — most tools (calendar, email, tasks, spending, habits, notes, finance, weather, messaging) are *not* pre-fetched. The LLM sees their schemas, decides mid-response whether to call one, the executor runs it, and the result is fed back as a `tool_result` block. See [Tool System](#tool-system).

4. **Stream response** — tokens go to the user as they arrive. The user starts reading while the model is still generating.

5. **Writeback last** — memory updates happen *after* the user has their response. It's `asyncio.create_task()`, not `await` — the user never waits for embedding, fact extraction, or cloud backup sync.

---

## Six-Layer Architecture

```
Input channels     voice · telegram · webui · cron
      ↓
Gateway            authenticate · resolve session · normalise to KairosEvent
      ↓
Orchestrator       classify request → pick tier → assemble context → tool loop → stream response
      ↓
Memory             SQLite (tasks/events/habits/spending/notes) · cosine-similarity vector store · session JSON · R2 cloud backup
      ↓
Tools              search · calendar · email (read+write) · finance · weather · tasks · spending · habits · notes · messaging
      ↓
Output             TTS → speaker · Telegram sendMessage · WebSocket → browser · SMTP → email
```

Every request becomes a `KairosEvent` before anything else touches it. The orchestrator never knows which channel sent the message — only the content and session.

---

## Three-Tier Model Cascade

Every request is classified by the tier-1 model alias (fast, intended to be cheap) before any higher-tier call. See the [Interactive Diagrams (HTML)](architecture.html) or [PDF](kairos_architecture.pdf) for the full routing flow.

| Tier | Purpose | Default model (`litellm/config.yaml`) |
|------|---------|------------------------------------------|
| 1 | Classifier, formatting, trivial Q&A, write-back fact extraction | `gemini/gemini-3.1-flash-lite` |
| 2 | Most routing, calendar, web search synthesis, general chat | `gemini/gemini-3.1-flash-lite` |
| 3 | Complex reasoning, code, research, multi-step planning | `gemini/gemini-3.1-flash-lite` |

All LLM calls go through [LiteLLM](https://github.com/BerriAI/litellm) proxy. Models are swapped in `litellm/config.yaml` — no code changes needed. **The shipped config routes all three tiers to the same Gemini model** (with two API key slots per tier for basic rotation under rate limits) — there's no built-in cost/latency tiering out of the box. To get an actually-free local tier, point a `tier1`/`tier2` entry at an Ollama model (e.g. `ollama/qwen2.5:3b-instruct`) and run `ollama serve` alongside LiteLLM; see [Setup](SETUP.md#4-optional-start-ollama-for-a-local-tier).

**Failure behaviour:** If the classifier fails (timeout, bad JSON, model error), it defaults to tier 2 with no domains/tools flagged. The user gets a slightly more expensive/generic response, but never a crash.

---

## Resilience & Fallback Strategies

Kairos is designed to degrade gracefully instead of failing hard.

- Streaming fallback sequence: tier 3 -> tier 2 -> tier 1
- Tool-loop fallback sequence: each tool round retries lower tiers
- Session-history safety wrapper: history read errors become empty history
- Max tool-round degradation: if rounds are exhausted, fallback to plain generation
- Final user-safe output: return a safe fallback message when all recovery paths fail

Full detail in [Resilience](RESILIENCE.md).

### Tier Fallback Flow

```
Preferred tier from classifier
                |
                v
          Try Tier N
                |
      +-----+-----+
      |           |
 success      failure
      |           |
 stream      Try next lower tier
 output         |
                          v
                     Tier 3 -> 2 -> 1
                          |
               all tiers exhausted
                          |
                          v
               Return safe fallback message
```

---

## Error Classification & Recovery

| Failure class | Typical source | Recovery |
|---------------|----------------|----------|
| Transient network/timeout | LiteLLM or upstream API timeout | Retry with exponential backoff (`LLM_MAX_RETRIES`) |
| Classifier parse/output failure | Invalid JSON or malformed fields | Clamp/drop invalid fields; fallback to safe default route |
| Session-history read failure | File/IO/session-store issues | Continue request with empty history |
| Tool input/schema failure | Invalid tool arguments from model | Reject tool call and continue safely |
| Tool handler timeout | Slow backend/tool service | Return tool error string and continue response path |
| Exhausted tool rounds | Repeated tool calls without final answer | Degrade to plain generation with fallback tiers |

Resilience knobs are configured in `.env` and documented in [Setup](SETUP.md) and [Resilience](RESILIENCE.md).

---

## Classifier

The classifier (`runtime/orchestrator/classifier.py`) runs on every request that isn't caught by the chitchat fast-path regex, and outputs:

```json
{
  "intent": "question | task | reminder | memory | chitchat | search | code",
  "complexity": 1,
  "domains": ["tasks", "events"],
  "needs_external_tools": true,
  "tools_needed": ["tasks"],
  "tier": 2
}
```

| Field | Purpose |
|-------|---------|
| `intent` | What the user wants (for logging and future routing) |
| `complexity` | 1–3 depth of response needed |
| `domains` | Which memory stores / context blocks to fetch (only what's relevant) |
| `tools_needed` | Which tools the LLM is likely to need in the agentic loop |
| `tier` | Which model tier to use |

**Valid intents:** `question`, `task`, `reminder`, `memory`, `chitchat`, `search`, `code`

**Valid domains and tools are resolved dynamically from the tool registry** — `get_valid_domains()` and `get_valid_tools()` in `classifier.py` read straight from `tools/registry.py` (`get_eligibility()` and each tool's `domain` field) rather than a hardcoded list. This means every new registered tool automatically becomes classifiable without touching the classifier file. As of Mark 3 this resolves to all 11 registered tools (`web_search`, `send_message`, `google_calendar`, `check_gmail`, `gmail_actions`, `finance`, `weather`, `tasks`, `spending`, `habits`, `notes`) plus the fixed SQLite domains (`tasks`, `events`, `habits`, `spending`, `memory`).

Unknown values from the model are silently dropped — never passed downstream. A fast-path regex (`_CHITCHAT_RE`) catches obvious greetings/acknowledgements and skips the LLM call entirely, returning a fixed `chitchat` result at zero cost.

---

## Memory System

Four layers, each with a different purpose:

| Layer | Technology | Holds | When used |
|-------|------------|-------|-----------|
| **Always-on context** | `profile.md` + `preferences.json` | Name, tone, goals, habits, extracted facts | Every single prompt |
| **Structured data** | SQLite (`kairos.db`) | Tasks, events, habits, spending, notes | When classifier flags relevant domains, or the LLM calls the corresponding agentic tool |
| **Semantic recall** | SQLite + Python cosine similarity (`memory/vector_store.py`) | Conversation history embeddings + explicit notes (tagged by `source`) | When classifier flags `memory` domain, or the `notes` tool's `semantic_search` action runs |
| **Session history** | JSON files (`data/sessions/`) | Last N turns per conversation | Always — last 8 turns in prompt |

### Structured Data Schema

```sql
tasks    → id, title, due_date, status, project, priority, created_at
events   → id, title, start_time, end_time, location, notes, source
habits   → id, name, last_done, streak, target_frequency
         + habit_logs → id, habit_id, date   (source of truth for streaks)
spending → id, amount, category, merchant, date, notes, source, sms_id
budgets  → id, category, month, amount
notes    → id, title, body, tags, link_type, link_id, created_at, updated_at
         + notes_fts → FTS5 virtual table mirroring title/body for keyword search
```

`memory_embeddings` (vector store) and `memory_meta` are additional tables in the same `kairos.db` file, created by `memory/vector_store.py` and `memory/sqlite_store.py` respectively.

### Session Compaction

Sessions auto-compact at 20 turns (`SESSION_MAX_TURNS` in `.env`). Old turns are summarised by the tier-1 model and stored; the last 2 turns are kept verbatim so there's always immediate context. This prevents context from growing unbounded.

### Writeback Pipeline

After every response, a background task (`memory/writeback.py`) runs:

1. **Embed** — vectorise the conversation turn for future semantic search (skipped for `chitchat` intent to avoid memory pollution)
2. **Session append** — save the turn to the session JSON file, triggering compaction if needed
3. **Fact extraction** — tier-1 model extracts structured facts (preference/fact/goal) into `preferences.json`
4. **Cloud sync** (if R2 configured) — pushes the updated session file and `preferences.json` to Cloudflare R2 in the background

All of this is non-blocking — `asyncio.create_task()`, not `await`.

---

## Tool System

As of Mark 3, Kairos registers **11 tools** in `runtime/tools/registry.py`. Full per-tool reference (parameters, actions, examples, failure modes) lives in the **[Tool Guide](TOOL_GUIDE.md)** — this section covers the dispatch mechanics shared by all of them.

### Two execution patterns

| Pattern | Tools | When it runs |
|---------|-------|---------------|
| **Pre-LLM** | `web_search` (the only entry in `PRE_LLM_TOOLS`) | Classifier flags it → orchestrator runs it before the first LLM call → results are injected into the system prompt as context |
| **Agentic** | `send_message`, `google_calendar`, `check_gmail`, `gmail_actions`, `finance`, `weather`, `tasks`, `spending`, `habits`, `notes` | LLM sees the schema via `get_tool_schemas()`, decides mid-response to call it, the executor runs it, results feed back as a `tool_result` block, looping up to `MAX_TOOL_ROUNDS` (default 5) |

### Web Search backends

Pluggable backend design — switch by setting `SEARCH_BACKEND` in `.env`:

| Backend | Free Tier | Key Required |
|---------|-----------|-------------|
| DuckDuckGo | Unlimited | No |
| Brave | 2000/month | Yes |
| Tavily | 1000/month | Yes |
| Serper | 2500 (no expiry) | Yes |

All backends return the same shape (`title`, `body`, `url`) — the formatter produces identical, citation-ready prompt blocks regardless of backend, with retry/backoff and a per-backend circuit breaker.

### Tool Validation

Tool inputs are validated against a JSON schema (in each tool's `REGISTRY` entry) before execution via `tools/executor.py`. The LLM's raw output never goes directly to a tool — eligibility (enabled + required env vars present) is checked, then the schema, then the handler is dispatched with a per-call timeout (`TOOL_TIMEOUT_SECONDS`, 15s). Every tool returns a plain string; none of them raise to the LLM.

---

## Observability (Langfuse)

Every request is wrapped in a Langfuse trace, propagated through `langfuse_client.start_observation()` calls in `orchestrator/orchestrator.py`:

- Root span: `kairos-process` (input text, channel, output)
- Child span: `classifier` (classification input/output)
- Child span: `context-assembly` (domains, tools needed, context/pretools/history sizes)
- Per-round generation spans during the tool loop and final streaming response, tagged with `generation_name` (e.g. `tool_round_0`, `direct_stream_response`, `fallback_degraded_stream`)

`trace_id` and `parent_observation_id` are threaded through every LLM call (`llm/client.py` accepts `trace_id`/`metadata` on `complete()`, `stream()`, and `complete_with_tools()`) so nested spans in Langfuse correctly attribute to the parent trace. Configure via `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_HOST` — if unset, tracing calls fail silently and the rest of the system is unaffected.

`litellm/config.yaml` also wires `success_callback`/`failure_callback` to `langfuse` directly at the proxy level, so LiteLLM-side request/response/cost data is captured independently of the orchestrator's own spans.

---

## Cloud Backup (Cloudflare R2)

`runtime/utils/storage.py` (`R2StorageManager`) optionally syncs persistent state to a Cloudflare R2 bucket via `boto3`:

- **At startup** (`sync_down` / `sync_down_sessions`, called from `main.py` before channels open): downloads `kairos.db`, `preferences.json`, `profile.md`, and all `sessions/*.json` files, so a fresh deploy (e.g. moving to a new VPS) picks up where the last one left off
- **In the background** (`sync_up_background`, called from `writeback.py` after every response): pushes the updated session file and `preferences.json` back up, fire-and-forget via `asyncio.create_task()`

Entirely opt-in — if `S3_ENDPOINT_URL`/`S3_ACCESS_KEY_ID`/`S3_SECRET_ACCESS_KEY`/`S3_BUCKET_NAME` aren't all set, `R2StorageManager.enabled` is `False` and every sync call becomes a no-op. See [Setup → Cloud Backup](SETUP.md#cloud-backup-cloudflare-r2-optional).

---

## Design Rules

These constraints are non-negotiable — they define what Kairos is:

| Rule | Rationale |
|------|-----------|
| Every response must stream | Never wait for full generation before output begins |
| LLM calls always go through LiteLLM | Never call Anthropic, OpenAI, or Gemini directly |
| Tool inputs validated against schema | Never pass raw LLM output to a tool |
| All I/O is async | Never block the event loop |
| Memory write-back is non-blocking | `asyncio.create_task()`, not `await` |
| Sessions compact automatically | Never grow unbounded |
| Telegram is the only proactive push channel | Voice/WebUI may not be active when cron fires; email handles the scheduled morning briefing instead |
| Classifier domains/tools are derived from the registry, not hardcoded | A new tool becomes classifiable the moment it's registered |

---

## Latency Targets

| Stage | Target | Tool |
|-------|--------|------|
| STT transcription | ~280ms | Deepgram streaming |
| Context assembly | <50ms | SQLite parallel fetch |
| Tier 1 first token | ~50–300ms | Depends on whether tier1 is routed local (Ollama) or cloud (Gemini, default) |
| Tier 2 first token | ~150–300ms | Same — see [Three-Tier Model Cascade](#three-tier-model-cascade) |
| Tier 3 first token | ~300ms | Gemini (default for all tiers) |
| TTS first chunk | ~80ms | Cartesia |
| **Total (Tier 1)** | **<500ms** | end of speech to first audio, local-routed |
| **Total (Tier 2)** | **<700ms** | end of speech to first audio |
| **Total (Tier 3)** | **<1200ms** | end of speech to first audio |

Use `runtime/latency_probe.py` to measure actual per-tier latency against your own `litellm/config.yaml` rather than relying on these targets — they assume a local tier-1/tier-2 route that the shipped config doesn't use by default.

---

## Deployment Phases

| Phase | Platform | Notes |
|-------|----------|-------|
| Dev | Laptop | All services local, Docker Compose optional |
| Always-on | Hetzner CX22 VPS (~€4/month) | 24/7 Telegram polling + cron |
| Self-hosted | Synology / TrueNAS NAS | Full data sovereignty — migrate by rsync of `data/`, or enable R2 sync for automatic cloud backup |