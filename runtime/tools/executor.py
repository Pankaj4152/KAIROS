"""
Tool executor — safe dispatch with schema validation, eligibility gating, and timeout.

Flow for every tool call:
    1. Check eligibility map (env vars present, tool enabled) — fast reject
    2. Validate input against JSON schema — reject malformed LLM output
    3. Load handler via lazy import — isolates import errors
    4. Dispatch with timeout — hung network calls don't block the response
    5. Return string result or error string — never raises
"""

import asyncio
import inspect
import logging
import time

from jsonschema import validate, ValidationError

from tools.registry import REGISTRY, get_eligibility

logger = logging.getLogger(__name__)

TOOL_TIMEOUT_SECONDS = 15


async def execute(tool_name: str, tool_input: dict) -> str:
    """
    Dispatch a tool call safely.

    Returns a string in all cases — either the tool result or an error message.
    Never raises. Caller can inject the return value directly into the prompt.
    """

    # ── 1. known tool? ─────────────────────────────────────────────────────────
    if tool_name not in REGISTRY:
        logger.warning("Rejected unknown tool: %r", tool_name)
        return f"Error: tool '{tool_name}' is not available"

    # ── 2. eligible? (enabled + env vars present) ──────────────────────────────
    eligibility = get_eligibility()
    if not eligibility.get(tool_name, False):
        tool = REGISTRY[tool_name]
        if not tool["enabled"]:
            logger.warning("Rejected disabled tool: %r", tool_name)
            return f"Error: tool '{tool_name}' is currently disabled"
        else:
            logger.warning("Rejected ineligible tool: %r (missing env vars)", tool_name)
            return f"Error: tool '{tool_name}' is not configured on this system"

    # ── 3. validate input schema ───────────────────────────────────────────────
    schema = REGISTRY[tool_name]["schema"]
    try:
        validate(instance=tool_input, schema=schema)
    except ValidationError as e:
        logger.warning("Tool %r input validation failed: %s", tool_name, e.message)
        return f"Error: invalid input for '{tool_name}': {e.message}"

    # ── 4. load handler (lazy import) ──────────────────────────────────────────
    try:
        loader = REGISTRY[tool_name]["handler"]
        handler = loader()           # each handler entry is a zero-arg factory fn
    except Exception as e:
        logger.error("Failed to load handler for tool %r: %s", tool_name, e)
        return f"Error: could not load tool '{tool_name}'"

    # ── 5. dispatch ────────────────────────────────────────────────────────────
    logger.info("Tool exec start  tool=%s inputs=%s", tool_name, list(tool_input.keys()))
    t0 = time.perf_counter()

    try:
        if inspect.iscoroutinefunction(handler):
            result = await asyncio.wait_for(
                handler(**tool_input),
                timeout=TOOL_TIMEOUT_SECONDS,
            )
        else:
            # sync handler — run in thread so we don't block the event loop
            result = await asyncio.wait_for(
                asyncio.to_thread(handler, **tool_input),
                timeout=TOOL_TIMEOUT_SECONDS,
            )

        logger.info(
            "Tool exec done   tool=%s duration=%.2fs chars=%d",
            tool_name, time.perf_counter() - t0, len(str(result)),
        )
        return str(result)

    except asyncio.TimeoutError:
        logger.warning(
            "Tool exec TIMEOUT  tool=%s after %ss",
            tool_name, TOOL_TIMEOUT_SECONDS,
        )
        return f"Error: tool '{tool_name}' timed out"

    except Exception as e:
        logger.warning(
            "Tool exec FAILED  tool=%s duration=%.2fs error=%s",
            tool_name, time.perf_counter() - t0, e,
        )
        return f"Error running '{tool_name}': {e}"