# Contributing

> How to extend Kairos — add channels, tools, search backends, and memory domains.

**← [Back to README](../README.md)** · [User Guide](GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Setup](SETUP.md)

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

from gateway.normalizer import normalize_generic, KairosEvent
from orchestrator.orchestrator import orchestrator

logger = logging.getLogger(__name__)


class YourChannel:

    async def _handle_message(self, raw_input: str, session_id: str) -> str:
        """Process a single message and return the full response."""
        event = KairosEvent(
            text=raw_input,
            channel="your_channel",
            session_id=session_id,
        )

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

### Register in `main.py`

```python
from channels.your_channel import YourChannel

# In main():
your_channel = YourChannel()
await asyncio.gather(
    telegram.run(),
    webui.run(),
    your_channel.run(),  # add here
)
```

That's it. Nothing else changes.

---

## Adding a New Tool

Tools provide external capabilities (search, calendar, messaging, etc.).

### Steps

1. Create `runtime/tools/your_tool.py` with an async function
2. Register it in `runtime/tools/registry.py`
3. Add the tool name to the classifier's valid tools
4. Add input mapping in the orchestrator

### 1. Create the tool

```python
"""
your_tool.py — brief description.
"""

import logging

logger = logging.getLogger(__name__)


async def your_tool(query: str) -> str:
    """
    Execute the tool and return a prompt-ready string.
    Must never raise — return error string on failure.
    """
    try:
        # Your tool logic here
        result = "..."
        return f"Tool result:\n{result}"
    except Exception as e:
        logger.warning("your_tool failed: %s", e)
        return f"Tool failed: {e}"
```

### 2. Register in `registry.py`

```python
from tools.your_tool import your_tool

TOOL_REGISTRY = {
    "web_search": web_search,
    "your_tool": your_tool,  # add here
}
```

### 3. Add to classifier's valid tools

In `runtime/orchestrator/classifier.py`:

```python
VALID_TOOLS = frozenset({"web_search", "calendar_write", "send_message", "your_tool"})
```

Also update the `CLASSIFIER_PROMPT` to tell the model when to use this tool.

### 4. Add input mapping in orchestrator

In `runtime/orchestrator/orchestrator.py`, inside `_run_tools()`:

```python
tool_inputs = {
    "web_search": {"query": query},
    "your_tool": {"query": query},  # add here
}
```

---

## Adding a Search Backend

Web search uses a pluggable backend system — all backends share the same interface.

### Steps

1. Subclass `SearchBackend` in `runtime/tools/web_search.py`
2. Implement `async def search(query, max_results) -> list[dict]`
3. Register in `_get_backend()`

### Template

```python
class YourBackend(SearchBackend):
    """
    Your search backend description.
    Free tier: X queries/month.
    Needs: YOUR_API_KEY in .env
    """

    def __init__(self):
        self.api_key = os.getenv("YOUR_API_KEY", "")

    async def search(self, query: str, max_results: int) -> list[dict]:
        if not self.api_key:
            logger.error("YOUR_API_KEY not set in .env")
            return []
        try:
            # Your search logic here
            # Must return: [{"title": ..., "body": ..., "url": ...}]
            return results
        except Exception as e:
            logger.warning("YourBackend search failed: %s", e)
            return []
```

### Register

```python
def _get_backend() -> SearchBackend:
    backends = {
        "duckduckgo": DuckDuckGoBackend,
        "brave":      BraveBackend,
        "tavily":     TavilyBackend,
        "serper":     SerperBackend,
        "yours":      YourBackend,  # add here
    }
```

Then set `SEARCH_BACKEND=yours` in `.env`.

---

## Adding a Memory Domain

Memory domains are structured data categories the classifier can flag for context injection.

### Steps

1. Add the table in `runtime/memory/sqlite_store.py`
2. Add a fetch function
3. Add a context block method in the orchestrator
4. Register the domain in the classifier

### 1. Add table to `init_db()`

```python
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

### 3. Add context block in orchestrator

In `runtime/orchestrator/orchestrator.py`:

```python
async def _fetch_your_domain_block(self) -> str | None:
    try:
        items = await fetch_your_domain()
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

### 4. Register in classifier

```python
VALID_DOMAINS = frozenset({"tasks", "events", "habits", "spending", "memory", "your_domain"})
```

Update the `CLASSIFIER_PROMPT` to tell the model when to flag this domain.

---

## Code Style

- **All I/O is async** — never block the event loop
- **SQLite runs in threads** — use `asyncio.to_thread()` for blocking calls
- **Never raise from tools** — return error strings, let the LLM handle gracefully
- **One singleton per module** — instantiate once, import everywhere
- **Validate all inputs** — never trust raw LLM output
- **Log warnings, not errors** — errors are for truly fatal things
