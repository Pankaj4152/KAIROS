"""
Orchestrator — the central request handler for Kairos.

Every message from every channel passes through Orchestrator.process().
Nothing else in the system should call the LLM directly.

Request flow:
    1. Classify  — Classifier decides tier, domains, intent (tier-1 local model)
    2. Build     — Assemble system prompt + context + history + user message
    3. Stream    — Yield response tokens as they arrive from LiteLLM
    4. Writeback — Save turns to session history (fire-and-forget background task)

What's not here yet (added in later steps):
    - Tool execution (web_search, calendar_write, etc.)
    - Vector memory search (sqlite-vec)
    - Full fact extraction writeback
"""

import asyncio
import logging
import os
from typing import AsyncGenerator

from dotenv import load_dotenv

load_dotenv()

from gateway.normalizer import KairosEvent
from llm.client import LLMClient
from memory.session_store import (
    append_turn,
    get_history,
    get_session,
    needs_compaction,
    compact,
)
from memory.sqlite_store import fetch_open_tasks, fetch_upcoming_events
from orchestrator.classifier import Classifier

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Routes every KairosEvent through classify → build → stream → writeback.

    Instantiate once as a module-level singleton.
    Inject a custom LLMClient in tests to avoid real API calls.
    """

    def __init__(self, llm_client: LLMClient | None = None, data_dir: str | None = None):
        self.llm        = llm_client or LLMClient()
        self.classifier = Classifier(llm_client=self.llm)
        self.data_dir   = data_dir or os.getenv("DATA_DIR", "./data")
        self._profile: str | None = None

    # ─── startup ──────────────────────────────────────────────────────────────

    async def startup(self) -> None:
        """
        Load profile from disk. Call once at app startup, not mid-request.
        Caches result in self._profile — subsequent calls are instant.
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

    async def _build_context(self, domains: list[str]) -> str:
        """
        Fetch ONLY the domains the classifier flagged, in parallel.

        Two 20ms SQLite queries in parallel = 20ms total.
        Two sequential = 40ms. At voice latency targets this matters.
        Fetching everything every request bloats the prompt unnecessarily.
        """
        coros = []

        if "tasks" in domains:
            coros.append(self._fetch_tasks_block())
        if "events" in domains:
            coros.append(self._fetch_events_block())

        if not coros:
            return ""

        results = await asyncio.gather(*coros, return_exceptions=True)

        blocks = [r for r in results if isinstance(r, str)]
        return "\n\n".join(blocks)

    async def _fetch_tasks_block(self) -> str | None:
        """Fetch open tasks and format as a context block."""
        try:
            tasks = await fetch_open_tasks()   # await — this is async
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
        """Fetch upcoming events and format as a context block."""
        try:
            events = await fetch_upcoming_events(limit=3)   # await — this is async
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

    # ─── prompt assembly ──────────────────────────────────────────────────────

    def _build_system_prompt(self, context: str) -> str:
        """
        Combine profile + context into a single system message.

        One system message is safer than two — some providers handle
        multiple system messages inconsistently.
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
        self, event: KairosEvent, domains: list[str]
    ) -> list[dict]:
        """
        Assemble the full messages list for the LLM call.

        Order:
            system  — profile + relevant context (always first, always one)
            history — last 8 turns including compaction summary if present
                      (get_history() injects the summary automatically)
            user    — the actual message (always last)
        """
        # Run context fetch and history load in parallel — both are I/O
        context, history = await asyncio.gather(
            self._build_context(domains),
            get_history(event.session_id, last_n=8),   # await handled by gather
        )

        messages = [
            {"role": "system", "content": self._build_system_prompt(context)},
            *history,
            {"role": "user", "content": event.text},
        ]

        return messages

    # ─── writeback ────────────────────────────────────────────────────────────

    async def _writeback(
        self,
        session_id: str,
        user_text: str,
        response_text: str,
        channel: str,
        tier: int,
    ) -> None:
        """
        Persist both turns to session history after response is delivered.
        Runs as a fire-and-forget background task — never blocks the response.

        What's not here yet:
            - Embedding the turn into sqlite-vec (vector_store.py)
            - Fact extraction (writeback.py in the memory layer)
        """
        try:
            await append_turn(
                session_id, "user", user_text,
                meta={"channel": channel},
            )
            await append_turn(
                session_id, "assistant", response_text,
                meta={"tier_used": tier},
            )

            if await needs_compaction(session_id):
                logger.info("Session %s needs compaction", session_id[:8])
                await self._compact_session(session_id)

        except Exception as e:
            logger.warning("Writeback failed for %s: %s", session_id[:8], e)

    async def _compact_session(self, session_id: str) -> None:
        """
        Summarise the full session history into 2-3 sentences using tier-1.
        compact() keeps the last 2 turns — session is never fully cold.
        """
        history = await get_history(session_id, last_n=20)
        if not history:
            return

        history_text = "\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in history
        )
        prompt = (
            "Summarise this conversation in 2-3 sentences. "
            "Focus on facts, decisions, and context useful for continuing later.\n\n"
            f"{history_text}"
        )

        summary = await self.llm.complete(
            [{"role": "user", "content": prompt}],
            tier=1,
            timeout=10.0,   # tighter timeout — this is background work
        )

        await compact(session_id, summary)
        logger.info("Session %s compacted", session_id[:8])

    # ─── main entry point ─────────────────────────────────────────────────────

    async def process(self, event: KairosEvent) -> AsyncGenerator[str, None]:
        """
        Process a KairosEvent end to end and stream the response.

        Flow:
            1. Classify  — what is this? which tier? which domains?
            2. Build     — system + context + history + user, in parallel where possible
            3. Stream    — yield tokens as they arrive from LiteLLM
            4. Writeback — save turns in background (non-blocking)

        The caller (voice.py, telegram.py, webui.py) just iterates the generator.
        It never needs to know about tiers, domains, or memory.
        """
        # Ensure profile is loaded — no-op after first call
        await self.startup()

        # Step 1 — classify
        classification = await self.classifier.classify(event.text)
        tier    = classification.get("tier", 2)
        domains = classification.get("domains", [])

        logger.debug(
            "Classified %r → tier=%s domains=%s",
            event.text[:40], tier, domains,
        )

        # Step 2 — build prompt (context + history fetched in parallel inside)
        messages = await self._build_messages(event, domains)

        # Step 3 — stream response, collect full text for writeback
        full_response: list[str] = []
        async for token in self.llm.stream(messages, tier=tier):
            full_response.append(token)
            yield token

        # Step 4 — writeback in background, never blocks the caller
        asyncio.create_task(
            self._writeback(
                event.session_id,
                event.text,
                "".join(full_response),
                event.channel,
                tier,
            )
        )


# ─── singleton ────────────────────────────────────────────────────────────────

# Import and use this everywhere. Call await orchestrator.startup() in main.py.
orchestrator = Orchestrator()