"""
Orchestrator — the central request handler for Kairos.

Every message from every channel passes through Orchestrator.process().
Nothing else in the system should call the LLM directly.

Request flow:
    1. Classify  — Classifier decides tier, domains, intent (tier-1 local model)
    2. Build     — Assemble system prompt + context + history + user message
    3. Tool loop — LLM may request tool calls; execute and feed results back
    4. Stream    — Yield final response tokens as they arrive from LiteLLM
    5. Writeback — embed + session append + fact extraction (fire-and-forget)

Two tool patterns:
    Pre-LLM tools  — web_search: classifier flags these,
                     results injected as context BEFORE the LLM call.
                     LLM sees the data already in the prompt.

    Agentic tools  — web_search etc:
                     LLM sees tool schemas, decides mid-response to call one,
                     we execute it, feed result back, LLM gives final answer.
                     This is the standard tool-use / function-calling loop.
"""

import asyncio
import json
import logging
import os
import time
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
from tools.registry import get_tool_schemas

logger = logging.getLogger(__name__)

# Tools the classifier pre-fetches before the LLM call.
# Everything else goes through the agentic loop.
PRE_LLM_TOOLS = {"web_search"}

# Max tool-call rounds before we stop looping.
# Prevents infinite loops if the LLM keeps requesting tools.
MAX_TOOL_ROUNDS = 5


class Orchestrator:
    """
    Routes every KairosEvent through classify → build → tool loop → stream → writeback.

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
            coros.append(search_as_context(query_text, top_k=5))

        if not coros:
            return ""

        results = await asyncio.gather(*coros, return_exceptions=True)
        blocks = [r for r in results if isinstance(r, str) and r.strip()]
        return "\n\n".join(blocks)

    async def _fetch_tasks_block(self) -> str | None:
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

    # ─── pre-LLM tools ────────────────────────────────────────────────────────

    async def _run_pre_llm_tools(self, tools_needed: list[str], query: str) -> str:
        """
        Run tools that fetch context BEFORE the LLM call.
        Results get injected into the system prompt as context.

        Only PRE_LLM_TOOLS are handled here. Agentic tools (write/update/delete)
        are handled in the tool loop after the first LLM call.
        """
        pre_tools = [t for t in tools_needed if t in PRE_LLM_TOOLS]
        if not pre_tools:
            return ""

        tool_inputs = {
            "web_search":          {"query": query}
        }

        coros, names = [], []
        for tool_name in pre_tools:
            if tool_name in tool_inputs:
                coros.append(execute(tool_name, tool_inputs[tool_name]))
                names.append(tool_name)
            else:
                logger.warning("No input mapping for pre-LLM tool: %s", tool_name)

        if not coros:
            return ""

        results = await asyncio.gather(*coros, return_exceptions=True)

        blocks = []
        for name, result in zip(names, results):
            if isinstance(result, Exception):
                logger.warning("Pre-LLM tool %s raised: %s", name, result)
            elif isinstance(result, str) and result.strip():
                blocks.append(result)

        return "\n\n".join(blocks)

    # ─── agentic tool loop ────────────────────────────────────────────────────

    async def _run_tool_loop(
        self,
        messages: list[dict],
        tier: int,
    ) -> AsyncGenerator[str, None]:
        """
        Agentic tool-use loop (Generator).
        Yields strings (tokens) as they arrive from the LLM.

        Final messages are available via current_messages (if needed).
        """
        from llm.client import STREAM_TIMEOUT
        response_timeout = STREAM_TIMEOUT

        tool_schemas = get_tool_schemas()

        # Filter to only agentic tools — pre-LLM tools already ran
        agentic_schemas = [
            s for s in tool_schemas
            if s["name"] not in PRE_LLM_TOOLS
        ]

        # If no agentic tools are available, real stream immediately
        if not agentic_schemas:
            async for token in self.llm.stream(
                messages, tier=tier, timeout=response_timeout,
            ):
                yield token
            return

        current_messages = list(messages)
        final_text = ""

        # For tool rounds, we currently use complete() because tool blocks 
        # need to be parsed as a whole. But we could technically stream the 
        # final round after all tool results are in.
        for round_num in range(MAX_TOOL_ROUNDS):
            response = await self.llm.complete_with_tools(
                messages=current_messages,
                tools=agentic_schemas,
                tier=tier,
                timeout=response_timeout,
            )
            
            tool_calls = [
                block for block in response.get("content", [])
                if block.get("type") == "tool_use"
            ]

            if not tool_calls:
                # No tool calls — LLM gave a final text response.
                # Fake stream it for the loop interface.
                text = _extract_text(response)
                for i in range(0, len(text), 10):
                    yield text[i:i+10]
                break

            # Append LLM's tool-calling response to messages
            current_messages.append({
                "role": "assistant",
                "content": response.get("content", []),
            })

            # Execute each tool call and collect results
            tool_results = []
            for call in tool_calls:
                tool_name  = call["name"]
                tool_input = call["input"]
                tool_id    = call["id"]

                logger.info(
                    "Agentic tool call [round %d]: %s inputs=%s",
                    round_num + 1, tool_name, list(tool_input.keys()),
                )

                result = await execute(tool_name, tool_input)

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tool_id,
                    "content":     result,
                })

            # Feed tool results back to LLM as a user message
            # (Anthropic API requires tool results in a user turn)
            current_messages.append({
                "role":    "user",
                "content": tool_results,
            })

            logger.debug("Tool loop round %d complete, looping back", round_num + 1)

        else:
            # Hit MAX_TOOL_ROUNDS without getting a final response.
            # Generator should yield tokens, not return them.
            logger.warning("Tool loop hit MAX_TOOL_ROUNDS=%d, forcing stop", MAX_TOOL_ROUNDS)
            yield "I ran into an issue completing that request. Please try again."

    # ─── prompt assembly ──────────────────────────────────────────────────────

    def _build_system_prompt(self, context: str) -> str:
        profile = self._profile or "You are Kairos, a personal AI assistant."

        base = f"""You are Kairos, a personal AI assistant.

{profile}

Guidelines:
- Be concise. The user may be listening via voice.
- Never start with filler like "Great!" or "Of course!".
- Respond in plain text ONLY. ABSOLUTELY NO markdown headers (#), bold (**), or bullet symbols (-/+) unless explicitly asked for formatting.
- If you don't know something, say so directly. (You are a helpful assistant, but value brevity above all else.)"""

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
        Assemble the full messages list for the first LLM call.
        Context fetch, pre-LLM tools, and history all run in parallel.
        """
        context_task = self._build_context(domains, event.text)
        tools_task   = self._run_pre_llm_tools(tools_needed, event.text)
        history_task = get_history(event.session_id, last_n=8)

        context, tool_results, history = await asyncio.gather(
            context_task,
            tools_task,
            history_task,
        )

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
            1. Classify  — tier, domains, tools via local phi-3-mini
            2. Build     — system + context + pre-LLM tools + history (parallel)
            3. Tool loop — agentic tools: LLM calls tools, we execute, loop back
            4. Stream    — yield final response tokens
            5. Writeback — embed + session + facts in background
        """
        t0 = time.perf_counter()
        sid = event.session_id[:8]
        logger.info(
            "REQ START  session=%s channel=%s text=%r",
            sid, event.channel, event.text[:60],
        )

        await self.startup()

        # Step 1 — classify
        t_classify = time.perf_counter()
        classification = await self.classifier.classify(event.text)
        tier         = classification.get("tier", 2)
        domains      = classification.get("domains", [])
        tools_needed = classification.get("tools_needed", [])

        logger.info(
            "REQ CLASSIFY  session=%s tier=%s domains=%s tools=%s duration=%.2fs",
            sid, tier, domains, tools_needed, time.perf_counter() - t_classify,
        )

        # Step 2 — build initial messages
        messages = await self._build_messages(event, domains, tools_needed)
        logger.debug(
            "REQ BUILD  session=%s messages=%d",
            sid, len(messages),
        )

        # Step 3 — agentic tool loop (Generator)
        # Yields tokens as they arrive.
        t_tools = time.perf_counter()
        full_text = ""
        async for token in self._run_tool_loop(messages, tier):
            full_text += token
            yield token

        logger.info(
            "REQ DONE  session=%s total=%.2fs chars=%d",
            sid, time.perf_counter() - t0, len(full_text),
        )

        # Step 4 — writeback
        asyncio.create_task(
            run_writeback(
                session_id=event.session_id,
                user_text=event.text,
                response_text=full_text,
                channel=event.channel,
                tier=tier,
                intent=classification.get("intent", "question"),
                llm=self.llm,
                data_dir=self.data_dir,
            )
        )


# ─── helpers ──────────────────────────────────────────────────────────────────

def _today() -> str:
    """Today's date in YYYY-MM-DD format."""
    from datetime import date
    return date.today().strftime("%Y-%m-%d")


def _extract_text(response: dict) -> str:
    """Pull plain text out of an Anthropic-style response dict."""
    parts = []
    for block in response.get("content", []):
        if block.get("type") == "text":
            parts.append(block["text"])
    return "".join(parts)


def _stream_text(text: str, chunk_size: int = 10):
    """
    Yield text in small chunks to simulate streaming.
    Channels expect a generator — this gives them one even when we
    already have the full response from the tool loop.
    """
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


# ─── singleton ────────────────────────────────────────────────────────────────

orchestrator = Orchestrator()