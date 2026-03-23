"""
Orchestrator — the central request handler for Kairos.

Every message from every channel passes through Orchestrator.process().
Nothing else in the system should call the LLM directly.

Request flow:
    1. Classify  — Classifier decides tier, domains, intent (tier-1 local model)
    2. Build     — Assemble system prompt + context + history + user message
    3. Stream    — Yield response tokens as they arrive from LiteLLM
    4. Writeback — embed + session append + fact extraction (fire-and-forget)

Memory domains the context assembler handles:
    tasks, events, habits, spending → sqlite_store.py (structured SQL)
    memory                          → vector_store.py (semantic search)
"""

import asyncio
import logging
import os
from typing import AsyncGenerator

from dotenv import load_dotenv

load_dotenv()

from gateway.normalizer import KairosEvent
from llm.client import LLMClient
from memory.session_store import get_history
from memory.writeback import run_writeback
from memory.vector_store import search_as_context
from memory.sqlite_store import (
    fetch_open_tasks,
    fetch_upcoming_events,
    fetch_habits,
    fetch_spending_summary,
)
from orchestrator.classifier import Classifier

from tools.executor import execute

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Routes every KairosEvent through classify → build → stream → writeback.

    Instantiate once as a module-level singleton.
    Inject a custom LLMClient in tests to avoid real API calls.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        data_dir: str | None = None,
    ):
        self.llm        = llm_client or LLMClient()
        self.classifier = Classifier(llm_client=self.llm)
        self.data_dir   = data_dir or os.getenv("DATA_DIR", "./data")
        self._profile: str | None = None

    # ─── startup ──────────────────────────────────────────────────────────────

    async def startup(self) -> None:
        """
        Load profile.md from disk once at app startup.
        Cached in self._profile — all subsequent calls are instant no-ops.
        Never call this mid-request.
        """
        if self._profile is not None:
            return
        path = os.path.join(self.data_dir, "profile.md")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self._profile = f.read()
        else:
            self._profile = "You are Kairos, a personal AI assistant."
        logger.debug("Profile loaded (%d chars)", len(self._profile))

    # ─── context assembly ─────────────────────────────────────────────────────

    async def _build_context(self, domains: list[str], query_text: str) -> str:
        """
        Fetch ONLY the domains the classifier flagged, in parallel.

        query_text is passed to vector search — it's the user's message,
        used to find semantically relevant past turns.

        Why parallel:
            Each domain fetch is independent I/O. asyncio.gather() runs
            them all at once — total time = slowest single fetch, not sum.
        """
        coros = []

        if "tasks" in domains:
            coros.append(self._fetch_tasks_block())
        if "events" in domains:
            coros.append(self._fetch_events_block())
        if "habits" in domains:
            coros.append(self._fetch_habits_block())
        if "spending" in domains:
            coros.append(self._fetch_spending_block())
        if "memory" in domains:
            # Vector search — finds semantically similar past turns
            coros.append(search_as_context(query_text, top_k=5))

        if not coros:
            return ""

        results = await asyncio.gather(*coros, return_exceptions=True)

        # Filter out None, empty strings, and exceptions
        # Exceptions are already logged inside each fetch method
        blocks = [r for r in results if isinstance(r, str) and r.strip()]
        return "\n\n".join(blocks)

    async def _fetch_tasks_block(self) -> str | None:
        """Open tasks formatted as a prompt context block."""
        try:
            tasks = await fetch_open_tasks()
            if not tasks:
                return None
            lines = "\n".join(
                f"  - [{t['priority']}] {t['title']}"
                f"{' (due ' + t['due_date'] + ')' if t['due_date'] else ''}"
                for t in tasks[:5]
            )
            return f"Open tasks:\n{lines}"
        except Exception as e:
            logger.warning("fetch_tasks failed: %s", e)
            return None

    async def _fetch_events_block(self) -> str | None:
        """Upcoming events formatted as a prompt context block."""
        try:
            events = await fetch_upcoming_events(limit=3)
            if not events:
                return None
            lines = "\n".join(
                f"  - {e['title']} at {e['start_time']}"
                for e in events
            )
            return f"Upcoming events:\n{lines}"
        except Exception as e:
            logger.warning("fetch_events failed: %s", e)
            return None

    async def _fetch_habits_block(self) -> str | None:
        """Habits with streak info formatted as a prompt context block."""
        try:
            habits = await fetch_habits()
            if not habits:
                return None
            lines = "\n".join(
                f"  - {h['name']} (streak: {h['streak']}, "
                f"last: {h['last_done'] or 'never'})"
                for h in habits
            )
            return f"Habits:\n{lines}"
        except Exception as e:
            logger.warning("fetch_habits failed: %s", e)
            return None

    async def _fetch_spending_block(self) -> str | None:
        """Spending summary formatted as a prompt context block."""
        try:
            rows = await fetch_spending_summary()
            if not rows:
                return None
            lines = "\n".join(
                f"  - {r['category']}: {r['total']:.2f}"
                for r in rows[:5]
            )
            return f"Spending by category:\n{lines}"
        except Exception as e:
            logger.warning("fetch_spending failed: %s", e)
            return None


    # tools execution ──────────────────────────────────────────────────────
    async def _run_tools(self, tools_needed: list[str], query: str) -> str:
        """
        Execute all tools the classifier flagged, in parallel.
        Returns a formatted block ready for prompt injection.

        Why run tools before the LLM call:
            The LLM needs the search results IN the prompt to answer correctly.
            If we called tools after, the response would already be generated.
            Tool results are context, not post-processing.

        Why parallel:
            If classifier returns ["web_search", "calendar_read"], both
            can run at the same time. No reason to wait for one before starting
            the other — they're independent I/O operations.
        """
        if not tools_needed:
            return ""

        # Map tool name to the input it needs
        # web_search gets the user's query directly
        # Future tools will have their own input logic
        tool_inputs = {
            "web_search": {"query": query},
        }

        coros = []
        names = []
        for tool_name in tools_needed:
            if tool_name in tool_inputs:
                coros.append(execute(tool_name, tool_inputs[tool_name]))
                names.append(tool_name)
            else:
                logger.warning("No input mapping for tool: %s", tool_name)

        if not coros:
            return ""

        results = await asyncio.gather(*coros, return_exceptions=True)

        blocks = []
        for name, result in zip(names, results):
            if isinstance(result, Exception):
                logger.warning("Tool %s raised: %s", name, result)
            elif isinstance(result, str) and result.strip():
                blocks.append(result)

        return "\n\n".join(blocks)
    # ─── prompt assembly ──────────────────────────────────────────────────────

    def _build_system_prompt(self, context: str) -> str:
        """
        Combine profile + context into a single system message.

        Single system message is safer than multiple — some providers
        handle multiple system messages inconsistently.
        Context is appended only when there is something to inject.
        """
        profile = self._profile or "You are Kairos, a personal AI assistant."

        base = f"""You are Kairos, a personal AI assistant.

{profile}

Guidelines:
- Be concise. The user may be listening via voice.
- Never start with filler like "Great!" or "Of course!".
- Respond in plain text. No markdown headers or bullet symbols unless asked.
- If you don't know something, say so directly."""

        if context:
            return f"{base}\n\n--- Current context ---\n{context}"
        return base

    async def _build_messages(
        self,
        event: KairosEvent,
        domains: list[str],
        tools_needed: list[str],
    ) -> list[dict]:
        """
        Assemble the full messages list for the LLM call.

        Order:
            system  — profile + structured context + tool results
            history — last 8 turns
            user    — the actual message

        context fetch, tool execution, and history load all run in parallel —
        all three are independent I/O, none depends on the others.
        """
        context_task = self._build_context(domains, event.text)
        tools_task   = self._run_tools(tools_needed, event.text)
        history_task = get_history(event.session_id, last_n=8)

        context, tool_results, history = await asyncio.gather(
            context_task,
            tools_task,
            history_task,
        )

        # Combine context blocks
        combined = "\n\n".join(
            block for block in [context, tool_results] if block.strip()
        )

        return [
            {"role": "system", "content": self._build_system_prompt(combined)},
            *history,
            {"role": "user", "content": event.text},
        ]

    # ─── main entry point ─────────────────────────────────────────────────────

    async def process(self, event: KairosEvent) -> AsyncGenerator[str, None]:
        """
        Process a KairosEvent end to end and stream the response.

        Flow:
            1. Classify  — tier, domains, intent via local phi-3-mini
            2. Build     — system + context + history assembled in parallel
            3. Stream    — yield tokens as they arrive from LiteLLM
            4. Writeback — embed + session + facts in background (non-blocking)

        Channels iterate this generator directly. They never need to know
        about tiers, domains, memory, or tools — that's all here.
        """
        await self.startup()  # no-op after first call

        # Step 1 — classify
        classification = await self.classifier.classify(event.text)
        tier         = classification.get("tier", 2)
        domains      = classification.get("domains", [])
        tools_needed = classification.get("tools_needed", [])

        logger.debug(
            "Classified %r → tier=%s domains=%s tools=%s",
            event.text[:40], tier, domains, tools_needed,
        )

        
        # Step 2 — build prompt
        messages = await self._build_messages(event, domains, tools_needed)
        # Step 3 — stream response
        full_response: list[str] = []
        async for token in self.llm.stream(messages, tier=tier):
            full_response.append(token)
            yield token

        # Step 4 — writeback: embed + session append + fact extraction
        # Fire-and-forget — user already has their response
        asyncio.create_task(
            run_writeback(
                session_id=event.session_id,
                user_text=event.text,
                response_text="".join(full_response),
                channel=event.channel,
                tier=tier,
                llm=self.llm,         # shared client — one connection pool
                data_dir=self.data_dir,
            )
        )


# ─── singleton ────────────────────────────────────────────────────────────────

orchestrator = Orchestrator()