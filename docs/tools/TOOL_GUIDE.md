# Kairos Tool Guide

> Index of every tool in the registry. Each tool has its own reference page with parameters, actions, examples, and failure modes.

**← [Back to README](../README.md)** · [User Guide](GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Setup](SETUP.md) · [Contributing](CONTRIBUTING.md)

Tools are invoked by the LLM during the agentic tool loop (or, for `web_search`, pre-fetched by the orchestrator before the first LLM call — see [Architecture → Tool System](ARCHITECTURE.md#tool-system)). The classifier flags `tools_needed`; `tools/executor.py` validates inputs against the JSON schema in `tools/registry.py`, then dispatches. Every tool returns a plain string — none of them raise to the LLM.

---

## Registered tools

| Tool | File | Env vars required | What it's for |
|------|------|--------------------|----------------|
| [web_search](tools/web_search.md) | `tools/web_search.py` | none (DuckDuckGo default) | Current events, prices, anything that may have changed since training |
| [send_message](tools/send_message.md) | `tools/messaging.py` | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_USER_ID` | Proactively push a message to the user's Telegram |
| [google_calendar](tools/google_calendar.md) | `tools/google_calendar.py` | none (OAuth `token.json`/`credentials.json`) | Full CRUD on Google Calendar events |
| [check_gmail](tools/check_gmail.md) | `tools/gmail_check.py` | `GMAIL_USER`, `GMAIL_APP_PASSWORD` | Read-only inbox access via IMAP |
| [gmail_actions](tools/gmail_actions.md) | `tools/gmail_actions.py` | `GMAIL_USER`, `GMAIL_APP_PASSWORD` | Send, reply, forward, delete, archive, move email via SMTP/IMAP |
| [finance](tools/finance.md) | `tools/finance.py` | none (Yahoo Finance, keyless) | Stock/crypto quotes, history, ticker search |
| [weather](tools/weather.md) | `tools/weather.py` | none (Open-Meteo, keyless) | Current conditions, daily forecast, hourly forecast |
| [tasks](tools/tasks.md) | `tools/tasks.py` | none (local `kairos.db`) | Create/list/update/complete/delete/search to-dos |
| [spending](tools/spending.md) | `tools/spending.py` | none (local `kairos.db`) | Manual expense logging, budgets, category reports |
| [habits](tools/habits.md) | `tools/habits.py` | none (local `kairos.db`) | Habit streaks, checkins, 30-day consistency heatmap |
| [notes](tools/notes.md) | `tools/notes.py` | none (semantic search reuses `LITELLM_BASE_URL`) | Freeform notes with keyword + semantic search |

`send_message` is the only **write** path to the user outside of the normal response stream — everything else either reads data or mutates the user's own calendar/inbox/local stores.

---

## Two tool-execution patterns

| Pattern | Tools | When it runs |
|---------|-------|---------------|
| **Pre-LLM** | `web_search` | Classifier flags it → orchestrator runs it *before* the first LLM call → results are injected into the system prompt as context. The LLM never "calls" this tool directly; it just sees the results already in context. |
| **Agentic** | everything else | The LLM sees tool schemas via `tools/registry.py:get_tool_schemas()`, decides mid-response to call one, the executor runs it, the result is fed back as a `tool_result` block, and the LLM continues (up to `MAX_TOOL_ROUNDS`, default 5 — see [Resilience](RESILIENCE.md)). |

`PRE_LLM_TOOLS` in `runtime/orchestrator/orchestrator.py` is the literal set that controls this — currently just `{"web_search"}`.

---

## Tool contract rules

These apply to every tool in the registry, existing or new:

| Rule | Why |
|------|-----|
| Never raise from a tool | The LLM receives the return string directly; an uncaught exception breaks the tool loop |
| Return strings only | The executor casts to `str(result)` — return rich text, not dicts |
| Validate inside the handler too | The JSON schema catches shape errors; the handler catches semantic errors (e.g. negative amounts, invalid enums) |
| Keep handlers focused | One tool, one concern — `check_gmail` reads, `gmail_actions` writes; they don't overlap |
| Use `asyncio.to_thread()` for blocking I/O | Never block the event loop with IMAP, SQLite, or sync HTTP calls |
| Log warnings, not errors | Errors are for fatal startup failures — tool failures are expected and handled |
| Action-based tools document every action | If a tool has an `action` enum, every action needs its own row in the actions table, required params, and an example call |

---

## Adding a new tool

Full walkthrough in [CONTRIBUTING.md](CONTRIBUTING.md#adding-a-new-tool). Quick checklist:

1. **Write the handler** in `runtime/tools/your_tool.py` — async entrypoint, never raises, blocking I/O via `asyncio.to_thread()`.
2. **Add a lazy loader** in `runtime/tools/registry.py`:
   ```python
   def _load_your_tool():
       from tools.your_tool import your_function
       return your_function
   ```
3. **Register in `REGISTRY`** with `description`, `schema`, `handler`, `enabled`, `requires_env`.
4. The classifier picks up valid tool/domain names **dynamically** from the registry (`get_valid_tools()` / `get_valid_domains()` in `orchestrator/classifier.py`) — no separate `VALID_TOOLS` list to maintain by hand.
5. **Test with the executor directly**:
   ```python
   from tools.executor import execute
   import asyncio
   result = asyncio.run(execute("your_tool", {"param": "value"}))
   print(result)
   ```
6. **Add `docs/tools/your_tool.md`** following the format of the existing pages, and add a row to the table at the top of this file.

---

*Last updated: June 2026 — Mark 3 release. Reflects all 11 tools currently in `tools/registry.py`.*
