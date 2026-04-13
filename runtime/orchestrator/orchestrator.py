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

Resilience features:
    - Stream-tier fallback: streaming falls back tier 3 → 2 → 1
    - Tool-loop tier fallback: tool calling falls back per round
    - Session history wrapper: history errors don't block responses
    - Max rounds degradation: falls back to plain generation when max rounds hit
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
from llm.client import LLMClient, LLMError
from llm.debug import trace
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
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "5"))

# Safe fallback message when all recovery attempts fail
FALLBACK_MESSAGE = "I'm having difficulty responding right now. Please try again."


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

    # ─── startup ─────────────────────────────────────────────────────────────────

    async def startup(self) -> None:
        if self._profile is not None:
            trace("Orchestrator.startup profile already loaded")
            return
        path = os.path.join(self.data_dir, "profile.md")
        trace("Orchestrator.startup loading profile path=%s", path)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self._profile = f.read()
        else:
            self._profile = "You are Kairos, a personal AI assistant."
        logger.debug("Profile loaded (%d chars)", len(self._profile))

    # ─── context assembly ─────────────────────────────────────────────────────────

    async def _build_context(self, domains: list[str], query_text: str) -> str:
        trace("Orchestrator.build_context domains=%s query_preview=%r", domains, query_text[:120])
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
            trace("Orchestrator.build_context no domain coroutines")
            return ""

        results = await asyncio.gather(*coros, return_exceptions=True)
        blocks = [r for r in results if isinstance(r, str) and r.strip()]
        trace("Orchestrator.build_context gathered=%d non_empty_blocks=%d", len(results), len(blocks))
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

    # ─── pre-LLM tools ────────────────────────────────────────────────────────────

    async def _run_pre_llm_tools(self, tools_needed: list[str], query: str) -> str:
        """
        Run tools that fetch context BEFORE the LLM call.
        Results get injected into the system prompt as context.

        Only PRE_LLM_TOOLS are handled here. Agentic tools (write/update/delete)
        are handled in the tool loop after the first LLM call.
        """
        pre_tools = [t for t in tools_needed if t in PRE_LLM_TOOLS]
        trace("Orchestrator.pre_llm_tools requested=%s runnable=%s", tools_needed, pre_tools)
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
            trace("Orchestrator.pre_llm_tools no valid tool inputs")
            return ""

        results = await asyncio.gather(*coros, return_exceptions=True)

        blocks = []
        for name, result in zip(names, results):
            if isinstance(result, Exception):
                logger.warning("Pre-LLM tool %s raised: %s", name, result)
            elif isinstance(result, str) and result.strip():
                blocks.append(result)

        trace("Orchestrator.pre_llm_tools completed tools=%s blocks=%d", names, len(blocks))
        return "\n\n".join(blocks)

    # ─── stream with tier fallback (ENHANCEMENT #1) ────────────────────────────────

    async def _stream_with_tier_fallback(
        self,
        messages: list[dict],
        tier: int,
        timeout: float = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream response with automatic tier fallback.
        
        Attempts to stream at current tier, falls back to lower tiers on LLMError.
        Yields tokens as they arrive; if all tiers fail, yields safe fallback message.
        
        This provides graceful degradation:
          - Tier 3 (Gemini) down → Falls back to tier 2 (Llama)
          - Tier 2 (Ollama) down → Falls back to tier 1 (Phi)
          - Tier 1 down → Yields FALLBACK_MESSAGE
        
        Args:
            messages: OpenAI-format message list
            tier: Starting tier (3=cloud, 2=local llama, 1=local phi)
            timeout: Optional timeout override (uses LLM client default if None)
        """
        from llm.client import STREAM_TIMEOUT
        
        if timeout is None:
            timeout = STREAM_TIMEOUT
        
        # Try tiers in order: current tier, then downgrade to lower tiers
        tiers_to_try = []
        if tier >= 3:
            tiers_to_try = [3, 2, 1]
        elif tier == 2:
            tiers_to_try = [2, 1]
        else:
            tiers_to_try = [1]
        
        last_error: Exception | None = None
        trace("Orchestrator.stream_fallback start preferred_tier=%d sequence=%s", tier, tiers_to_try)
        
        for attempt_tier in tiers_to_try:
            try:
                logger.info(
                    "Stream attempt starting  tier=%d timeout=%.1fs",
                    attempt_tier, timeout,
                )
                t0 = time.perf_counter()
                token_count = 0
                
                async for token in self.llm.stream(
                    messages, tier=attempt_tier, timeout=timeout
                ):
                    yield token
                    token_count += 1
                
                logger.info(
                    "Stream succeeded  tier=%d tokens=%d duration=%.2fs",
                    attempt_tier, token_count, time.perf_counter() - t0,
                )
                trace("Orchestrator.stream_fallback success tier=%d tokens=%d", attempt_tier, token_count)
                return  # Success - exit immediately
                
            except LLMError as e:
                last_error = e
                trace("Orchestrator.stream_fallback failure tier=%d error=%s", attempt_tier, e)
                logger.warning(
                    "Stream failed at tier=%d: %s, attempting fallback",
                    attempt_tier, e,
                )
                
                # If this was the last tier, give up gracefully
                if attempt_tier == 1:
                    logger.error(
                        "Stream exhausted all tiers, last error: %s", last_error
                    )
                    trace("Orchestrator.stream_fallback exhausted tiers last_error=%s", last_error)
                    yield FALLBACK_MESSAGE
                    return
                
                # Otherwise, try next tier (loop continues)
                continue
        
        # Should not reach here, but just in case
        trace("Orchestrator.stream_fallback reached terminal fallback path")
        yield FALLBACK_MESSAGE

    # ─── safe session history wrapper (ENHANCEMENT #4) ─────────────────────────────

    async def _safe_get_history(
        self, session_id: str, last_n: int = 8
    ) -> list[dict]:
        """
        Fetch session history with graceful fallback to empty history.
        
        Session read errors (corrupted file, I/O error, etc.) do not block
        the response. The request continues with empty history, allowing the
        user to still get an answer.
        
        This prevents cascading failures where a corrupted session file
        causes the entire response pipeline to fail.
        
        Args:
            session_id: Which session to fetch
            last_n: How many turns to retrieve
        
        Returns:
            List of OpenAI-format message dicts, or empty list on error
        """
        try:
            history = await get_history(session_id, last_n=last_n)
            logger.debug("Session history loaded successfully: %d turns", len(history))
            trace("Orchestrator.safe_history loaded session=%s turns=%d", session_id[:8], len(history))
            return history
            
        except Exception as e:
            trace("Orchestrator.safe_history fallback empty session=%s error=%s", session_id[:8], e)
            logger.warning(
                "Failed to fetch session history for %s: %s, continuing without history",
                session_id[:8], e,
            )
            return []  # Empty history - request continues

    # ─── agentic tool loop with tier fallback (ENHANCEMENT #2 + #3) ────────────────

    async def _run_tool_loop(
        self,
        messages: list[dict],
        tier: int,
        tools_needed: list[str],
    ) -> AsyncGenerator[str, None]:
        """
        Agentic tool-use loop with tier fallback and graceful degradation.
        
        Flow:
            1. Try tool calling rounds (MAX_TOOL_ROUNDS attempts)
            2. Per round: try tiers 3→2→1 if tool-calling fails
            3. If max rounds hit: fall back to plain generation
            4. For plain generation: use stream with tier fallback
        
        Yields strings (tokens) as they arrive from the LLM.
        
        ENHANCEMENT #2: Tool round failures now gracefully downgrade tiers.
        ENHANCEMENT #3: Max rounds exhaustion falls back to plain response.
        """
        from llm.client import STREAM_TIMEOUT
        response_timeout = STREAM_TIMEOUT
        trace(
            "Orchestrator.tool_loop start tier=%d tools_needed=%s timeout=%.1f",
            tier,
            tools_needed,
            response_timeout,
        )

        tool_schemas = get_tool_schemas()

        # Filter to only agentic tools explicitly requested by the classifier.
        # This prevents the LLM from getting confused by unused tool schemas,
        # especially the tier-1 local model during simple chitchat.
        agentic_schemas = [
            s for s in tool_schemas
            if s["name"] not in PRE_LLM_TOOLS and s["name"] in tools_needed
        ]

        # If no agentic tools are available, stream immediately with tier fallback
        if not agentic_schemas:
            logger.debug("No agentic tools, streaming directly")
            trace("Orchestrator.tool_loop no_agentic_tools streaming_direct")
            async for token in self._stream_with_tier_fallback(
                messages, tier, timeout=response_timeout
            ):
                yield token
            return

        current_messages = list(messages)
        current_tier = tier  # Track which tier succeeded for tool calls

        # Tool calling loop with max rounds
        for round_num in range(MAX_TOOL_ROUNDS):
            response = None
            tool_round_succeeded = False
            trace("Orchestrator.tool_loop round=%d current_tier=%d", round_num + 1, current_tier)
            
            # ENHANCEMENT #2: Try tool calling at current tier, fall back if needed
            for attempt_tier in _get_tier_fallback_sequence(current_tier):
                try:
                    logger.info(
                        "Tool round %d: attempting complete_with_tools at tier=%d",
                        round_num + 1, attempt_tier,
                    )
                    
                    response = await self.llm.complete_with_tools(
                        messages=current_messages,
                        tools=agentic_schemas,
                        tier=attempt_tier,
                        timeout=response_timeout,
                    )
                    
                    current_tier = attempt_tier  # Remember successful tier
                    tool_round_succeeded = True
                    trace(
                        "Orchestrator.tool_loop round=%d tier=%d complete_with_tools ok",
                        round_num + 1,
                        attempt_tier,
                    )
                    logger.info("Tool round %d succeeded at tier=%d", round_num + 1, attempt_tier)
                    break  # Exit fallback loop
                    
                except LLMError as e:
                    trace(
                        "Orchestrator.tool_loop round=%d tier=%d failed error=%s",
                        round_num + 1,
                        attempt_tier,
                        e,
                    )
                    logger.warning(
                        "Tool round %d failed at tier=%d: %s",
                        round_num + 1, attempt_tier, e,
                    )
                    
                    if attempt_tier == 1:
                        # All tiers exhausted for this round
                        logger.error(
                            "Tool round %d: all tiers exhausted, degrading to plain generation",
                            round_num + 1,
                        )
                        # Fall through to degradation path below
                        break
            
            # If tool calling failed completely, degrade to plain generation
            if not tool_round_succeeded:
                trace("Orchestrator.tool_loop degrade_plain_generation round=%d", round_num + 1)
                logger.warning(
                    "Tool calling failed at all tiers in round %d, degrading to plain generation",
                    round_num + 1,
                )
                async for token in self._stream_with_tier_fallback(
                    current_messages, current_tier, timeout=response_timeout
                ):
                    yield token
                return
            
            # Parse tool calls from response
            tool_calls = [
                block for block in response.get("content", [])
                if block.get("type") == "tool_use"
            ]

            if not tool_calls:
                # No tool calls — LLM gave a final text response.
                text = _extract_text(response)  # response already has the right format
                trace("Orchestrator.tool_loop final_text round=%d chars=%d", round_num + 1, len(text))
                for i in range(0, len(text), 10):
                    yield text[i:i+10]
                logger.info("Tool loop completed with final text response at round %d", round_num + 1)
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
                trace(
                    "Orchestrator.tool_loop exec_tool round=%d name=%s keys=%s",
                    round_num + 1,
                    tool_name,
                    list(tool_input.keys()),
                )

                logger.info(
                    "Agentic tool call [round %d]: %s inputs=%s",
                    round_num + 1, tool_name, list(tool_input.keys()),
                )
                result = await execute(tool_name, tool_input)

                # Append a strict directive to the result. Local models like Llama 3 
                # often hallucinate additional tools after successful calls. This forces it
                # to exit the loop and reply to the user.
                result += (
                    "\n\n[SYSTEM DIRECTIVE: The tool has executed successfully. "
                    "You MUST now reply in plain text ONLY. DO NOT call any further tools.]"
                )

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tool_id,
                    "content":     result,
                })

            trace(
                "Orchestrator.tool_loop round=%d tool_calls=%d tool_results=%d",
                round_num + 1,
                len(tool_calls),
                len(tool_results),
            )

            # Feed tool results back to LLM as a user message
            # (Anthropic API requires tool results in a user turn)
            current_messages.append({
                "role":    "user",
                "content": tool_results,
            })

            logger.debug("Tool loop round %d complete, looping back", round_num + 1)

        else:
            # ENHANCEMENT #3: Hit MAX_TOOL_ROUNDS without getting a final response.
            # Fall back to plain generation instead of hard failure.
            trace("Orchestrator.tool_loop max_rounds_exhausted=%d", MAX_TOOL_ROUNDS)
            logger.warning(
                "Tool loop exhausted MAX_TOOL_ROUNDS=%d, degrading to plain generation",
                MAX_TOOL_ROUNDS,
            )
            
            try:
                async for token in self._stream_with_tier_fallback(
                    current_messages, current_tier, timeout=response_timeout
                ):
                    yield token
            except Exception as e:
                trace("Orchestrator.tool_loop final_plain_fallback_failed error=%s", e)
                logger.error("Final fallback also failed: %s", e)
                yield FALLBACK_MESSAGE

    # ─── prompt assembly ──────────────────────────────────────────────────────────

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
        
        Uses safe_get_history (ENHANCEMENT #4) so history errors don't block response.
        """
        context_task = self._build_context(domains, event.text)
        tools_task   = self._run_pre_llm_tools(tools_needed, event.text)
        history_task = self._safe_get_history(event.session_id, last_n=8)

        context, tool_results, history = await asyncio.gather(
            context_task,
            tools_task,
            history_task,
        )
        trace(
            "Orchestrator.build_messages context_chars=%d pretools_chars=%d history_turns=%d",
            len(context),
            len(tool_results),
            len(history),
        )

        combined = "\n\n".join(
            block for block in [context, tool_results] if block.strip()
        )

        return [
            {"role": "system", "content": self._build_system_prompt(combined)},
            *history,
            {"role": "user", "content": event.text},
        ]

    # ─── main entry point ─────────────────────────────────────────────────────────

    async def process(self, event: KairosEvent) -> AsyncGenerator[str, None]:
        """
        Process a KairosEvent end to end and stream the response.

        Flow:
            1. Classify  — tier, domains, tools via local phi-3-mini
            2. Build     — system + context + pre-LLM tools + history (parallel)
            3. Tool loop — agentic tools: LLM calls tools, we execute, loop back
            4. Stream    — yield final response tokens
            5. Writeback — embed + session + facts in background
        
        With resilience enhancements:
            - All streaming uses tier fallback
            - Tool loop tiers fall back on error
            - Max rounds degrade to plain generation
            - Session history errors don't block response
        """
        t0 = time.perf_counter()
        sid = event.session_id[:8]
        trace(
            "Orchestrator.process start session=%s channel=%s text_len=%d",
            sid,
            event.channel,
            len(event.text),
        )
        
        logger.info(
            "REQ START  session=%s channel=%s text=%r",
            sid, event.channel, event.text[:60],
        )
        logger.debug("Full request text: %r", event.text)
        
        await self.startup()
        
        # Step 1 — classify
        t_classify = time.perf_counter()
        logger.debug("Step 1: Starting classification...")
        classification = await self.classifier.classify(event.text)
        tier         = classification.get("tier", 2)
        domains      = classification.get("domains", [])
        tools_needed = classification.get("tools_needed", [])
        intent       = classification.get("intent", "unknown")
        trace(
            "Orchestrator.process classification session=%s intent=%s tier=%d domains=%s tools=%s",
            sid,
            intent,
            tier,
            domains,
            tools_needed,
        )
        
        logger.info(
            "REQ CLASSIFY  session=%s intent=%s tier=%d domains=%s tools=%s duration=%.2fs",
            sid, intent, tier, domains, tools_needed, time.perf_counter() - t_classify,
        )
        logger.debug("Full classification: %s", classification)
        
        # Step 2 — build messages
        logger.debug("Step 2: Building messages...")
        t_build = time.perf_counter()
        messages = await self._build_messages(event, domains, tools_needed)
        trace("Orchestrator.process built_messages session=%s count=%d", sid, len(messages))
        logger.debug("REQ BUILD  session=%s messages=%d duration=%.2fs", 
                    sid, len(messages), time.perf_counter() - t_build)
        for i, msg in enumerate(messages):
            logger.debug("  Message %d: role=%s, content_len=%d", 
                        i, msg.get("role"), len(str(msg.get("content", ""))))
        
        # Step 3 — tool loop
        logger.debug("Step 3: Starting tool loop...")
        t_tools = time.perf_counter()
        full_text = ""
        async for token in self._run_tool_loop(messages, tier, tools_needed):
            full_text += token
            yield token

        trace("Orchestrator.process response_complete session=%s chars=%d", sid, len(full_text))
        
        logger.info(
            "REQ DONE  session=%s total=%.2fs chars=%d classify=%.2fs build=%.2fs tools=%.2fs",
            sid, time.perf_counter() - t0, len(full_text), 
            time.perf_counter() - t_classify,
            t_build - t_classify,
            time.perf_counter() - t_tools,
        )
        
        # Step 4 — writeback
        logger.debug("Step 4: Starting writeback in background...")
        asyncio.create_task(
            run_writeback(
                session_id=event.session_id,
                user_text=event.text,
                response_text=full_text,
                channel=event.channel,
                tier=tier,
                intent=intent,
                llm=self.llm,
                data_dir=self.data_dir,
            )
        )
        trace("Orchestrator.process writeback_scheduled session=%s", sid)


# ─── helpers ──────────────────────────────────────────────────────────────────────

def _get_tier_fallback_sequence(tier: int) -> list[int]:
    """
    Return the sequence of tiers to try for fallback.
    
    If current tier is 3, try [3, 2, 1].
    If current tier is 2, try [2, 1].
    If current tier is 1, try [1].
    """
    if tier >= 3:
        return [3, 2, 1]
    elif tier == 2:
        return [2, 1]
    else:
        return [1]


def _today() -> str:
    """Today's date in YYYY-MM-DD format."""
    from datetime import date
    return date.today().strftime("%Y-%m-%d")


def _extract_text(response: dict) -> str:
    """Pull plain text out of an Anthropic-style response dict."""
    parts = []
    for block in response.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)

def _stream_text(text: str, chunk_size: int = 10):
    """
    Yield text in small chunks to simulate streaming.
    Channels expect a generator — this gives them one even when we
    already have the full response from the tool loop.
    """
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


# ─── singleton ───────────────────────────────────────────────────────────────────

orchestrator = Orchestrator()