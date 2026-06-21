# Contributing

> How to extend Kairos — add channels, tools, search backends, and memory domains.

**← [Back to README](../README.md)** · [User Guide](GUIDE.md) · [Tool Guide](TOOL_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Setup](SETUP.md) · [Diagrams HTML](architecture.html) · [Diagrams PDF](kairos_architecture.pdf)

---

## How Kairos Is Built

Before extending anything, understand the core flow:

```
Channel → Gateway → Orchestrator → Memory + Tools → LLM → Response
```

Each layer is independent. Channels don't know about memory. Tools don't know about channels. The orchestrator connects everything. This means you can add to one layer without touching any other.

---

## Adding a New Channel

A channel is any interface that accepts user input and delivers responses.

### Steps

1. Create `runtime/channels/your_channel.py`
2. Implement a class with a `run()` method
3. Import and register in `runtime/main.py`

### Template

```python
"""
YourChannel — brief description of what this channel does.
"""

import asyncio
import logging

from gateway.normalizer import normalize_webui  # or write your own normalize_* function
from orchestrator.orchestrator import orchestrator

logger = logging.getLogger(__name__)


class YourChannel:

    async def _handle_message(self, raw_input: str, session_id: str) -> str:
        """Process a single message and return the full response."""
        event = normalize_webui(raw_input, session_id)   # swap for a channel-specific normalizer

        tokens = []
        async for token in orchestrator.process(event):
            tokens.append(token)

        return "".join(tokens)

    async def run(self) -> None:
        """Start the channel. Runs until cancelled."""
        logger.info("YourChannel starting...")
        await orchestrator.startup()

        # Your event loop / listener here
        await asyncio.Event().wait()
```

> Every channel must produce a `KairosEvent` via one of the `normalize_*()` functions in `gateway/normalizer.py` — never construct a `KairosEvent` directly. Add a new `normalize_your_channel()` function there if none of the existing ones fit (see `normalize_telegram`, `normalize_voice`, `normalize_webui`, `normalize_cron` for the pattern).

### Register in `main.py`

```python
from channels.your_channel import YourChannel

# In main(), alongside telegram/webui:
your_channel = YourChannel()
await asyncio.gather(
    run_channel_safely("Telegram", telegram.run()),
    run_channel_safely("WebUI", webui.run()),
    run_channel_safely("YourChannel", your_channel.run()),  # add here
    return_exceptions=True,
)
```

`run_channel_safely()` catches fatal errors per-channel so one channel crashing doesn't take down the others — wrap new channels the same way.

That's it. Nothing else changes.

---

## Adding a New Tool

Tools provide external capabilities (search, calendar, messaging, finance, etc.). See the [Tool Guide](TOOL_GUIDE.md) for what's already registered before adding something that overlaps.

### Steps

1. Create `runtime/tools/your_tool.py` with an async function
2. Register it in `runtime/tools/registry.py`
3. Write `docs/tools/your_tool.md` and link it from `docs/TOOL_GUIDE.md`

That's the whole list — there's no separate classifier file to edit. `tools_needed` and `domains` are resolved dynamically from `REGISTRY` at classify time (see [Architecture → Classifier](ARCHITECTURE.md#classifier)), so a newly registered tool is automatically classifiable without touching `orchestrator/classifier.py`.

### 1. Create the tool

```python
"""
your_tool.py — brief description.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def your_tool(query: str) -> str:
    """
    Execute the tool and return a prompt-ready string.
    Must never raise — return an error string on failure.
    Blocking I/O goes through asyncio.to_thread().
    """
    try:
        # Your tool logic here
        result = "..."
        return f"Tool result:\n{result}"
    except Exception as e:
        logger.warning("your_tool failed: %s", e)
        return f"Error: your_tool failed — {e}"
```

### 2. Register in `registry.py`

Add a lazy loader (so a broken tool file can't crash the whole registry on import) and a `REGISTRY` entry:

```python
def _load_your_tool():
    from tools.your_tool import your_tool
    return your_tool


REGISTRY: dict[str, dict] = {
    # ...existing entries...

    "your_tool": {
        "domain": "your_domain",     # optional — feeds dynamic domain resolution
        "description": (
            "What this tool does and when the LLM should call it. "
            "Be specific — this is the only thing the LLM sees to decide when to use it."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "handler":      _load_your_tool,
        "enabled":      True,
        "requires_env": ["YOUR_API_KEY"],   # empty list if no env vars needed
    },
}
```

If `requires_env` lists any vars and they're missing at startup, `check_eligibility()` marks the tool ineligible and the executor rejects calls to it with a clear error — the tool simply won't be offered to the LLM. No further wiring is needed for the tool to become available once env vars are present.

### 3. (Agentic tools only) — nothing else to do

Most new tools are **agentic**: the LLM sees the schema, decides to call it, the executor dispatches it, and the result is fed back automatically by the tool loop in `orchestrator/orchestrator.py` — see `_run_tool_loop()`. You don't need to add input-mapping code anywhere.

Only **pre-LLM** tools (run *before* the first LLM call, results injected as context — currently just `web_search`) need an entry in the `tool_inputs` dict inside `_run_pre_llm_tools()` in `orchestrator/orchestrator.py`, plus adding the tool's name to the `PRE_LLM_TOOLS` set at the top of that file. Default to agentic unless you have a specific reason the LLM needs the data before it starts responding.

### 4. Test with the executor directly

```python
from tools.executor import execute
import asyncio
result = asyncio.run(execute("your_tool", {"query": "value"}))
print(result)
```

### 5. Document it

Add `docs/tools/your_tool.md` (use an existing page, e.g. `docs/tools/weather.md`, as a template) and add a row to the table at the top of `docs/TOOL_GUIDE.md`.

---

## Adding a Search Backend

Web search uses a pluggable backend system — all backends share the same interface.

### Steps

1. Subclass `SearchBackend` in `runtime/tools/web_search.py`
2. Implement `async def _fetch(self, query: str, max_results: int) -> list[SearchResult]`
3. Register in `BACKEND_MAP`

### Template

```python
class YourBackend(SearchBackend):
    """
    Your search backend description.
    Free tier: X queries/month.
    Needs: YOUR_API_KEY in .env
    """

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("YOUR_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError("YOUR_API_KEY not set in environment")

    async def _fetch(self, query: str, max_results: int) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(..., params={"q": query, "count": max_results})
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return [
                SearchResult(title=r["title"], body=r.get("snippet", ""), url=r["url"])
                for r in results
            ]
```

Retry/backoff, the circuit breaker, result validation/cleaning, and deduplication are all handled by the `SearchBackend` base class's `search()` method — you only implement the raw `_fetch()`.

### Register

```python
BACKEND_MAP: dict[str, type[SearchBackend]] = {
    "duckduckgo": DuckDuckGoBackend,
    "brave":      BraveBackend,
    "tavily":     TavilyBackend,
    "serper":     SerperBackend,
    "yours":      YourBackend,  # add here
}
```

Then set `SEARCH_BACKEND=yours` in `.env`. If your backend's `__init__` raises `EnvironmentError` (missing key), `_ensure_backends()` logs it and falls back to DuckDuckGo automatically.

---

## Adding a Memory Domain

Memory domains are structured data categories the classifier can flag for context injection.

### Steps

1. Add the table in `runtime/memory/sqlite_store.py`
2. Add a fetch function
3. Add a context block method in the orchestrator
4. Add the domain name to `get_valid_domains()` in `classifier.py` if it's not already covered by a tool's `domain` field

### 1. Add table to `init_db()`

```sql
CREATE TABLE IF NOT EXISTS your_domain (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    field1     TEXT NOT NULL,
    field2     TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### 2. Add fetch function

```python
def fetch_your_domain() -> list[dict]:
    def _run():
        with get_conn() as conn:
            rows = conn.execute("SELECT * FROM your_domain ...").fetchall()
            return [dict(r) for r in rows]
    return _run()
```

This is a plain sync function — callers must wrap it in `asyncio.to_thread()`, the same way `orchestrator.py`'s context-block methods already do for `fetch_open_tasks`, `fetch_upcoming_events`, etc.

### 3. Add context block in orchestrator

In `runtime/orchestrator/orchestrator.py`:

```python
async def _fetch_your_domain_block(self) -> str | None:
    try:
        items = await asyncio.to_thread(fetch_your_domain)
        if not items:
            return None
        lines = "\n".join(f"  - {item['field1']}" for item in items)
        return f"Your domain:\n{lines}"
    except Exception as e:
        logger.warning("fetch_your_domain failed: %s", e)
        return None
```

And add to `_build_context()`:

```python
if "your_domain" in domains:
    coros.append(self._fetch_your_domain_block())
```

### 4. Make the domain classifiable

`get_valid_domains()` in `orchestrator/classifier.py` already includes the fixed SQLite domains (`tasks`, `events`, `habits`, `spending`, `memory`) plus every `domain` value declared on a registered tool. If `your_domain` isn't backed by a tool, add it to the fixed set in `get_valid_domains()`:

```python
def get_valid_domains() -> set[str]:
    db_domains = {"tasks", "events", "habits", "spending", "memory", "your_domain"}
    tool_domains = {tool["domain"] for tool in REGISTRY.values() if tool.get("domain")}
    return db_domains | tool_domains
```

Also mention the domain in `get_classifier_prompt()`'s guidance so the model knows when to flag it — the valid-domains list itself is already injected dynamically, but the rules text may need a line explaining when this domain applies.

---

## Code Style

- **All I/O is async** — never block the event loop
- **SQLite runs in threads** — use `asyncio.to_thread()` for blocking calls
- **Never raise from tools** — return error strings, let the LLM handle gracefully
- **One singleton per module** — instantiate once, import everywhere
- **Validate all inputs** — never trust raw LLM output; JSON schema first, handler-level validation second
- **Log warnings, not errors** — errors are for truly fatal things; tool/handler failures are expected and recoverable
- **Destructive actions confirm first** — anything irreversible (e.g. `gmail_actions`' `delete_permanent`) should be designed so the LLM is expected to confirm with the user before calling it, and the tool's own docstring/description should say so explicitly