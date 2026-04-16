"""
LLM client — async wrapper around the LiteLLM proxy.

All LLM calls in Kairos go through this file. Nothing calls Anthropic,
OpenAI, or Ollama directly. Swap models by changing litellm/config.yaml,
not this file.

Two methods:
  stream()   — token-by-token generator. Use for all user-facing responses.
  complete() — single string result. Use for classifier, fact extractor,
               session compaction — anything internal where streaming adds no value.

Singleton:
  Import `llm` directly. One shared httpx connection pool for the whole app.
  Only instantiate LLMClient manually in tests.
"""

import asyncio
import json
import logging
import os
import time
from typing import AsyncGenerator

import httpx
from dotenv import load_dotenv
from llm.debug import trace, debug_payload, debug_messages

load_dotenv()

logger = logging.getLogger(__name__)


# ─── config ───────────────────────────────────────────────────────────────────

# Model names must match the model_name entries in litellm/config.yaml exactly.
# Change models there — not here.
TIER_MODELS = {
    1: "tier1",
    2: "tier2",
    3: "tier3",
}
# Single source of truth for all timeouts.
# stream() uses STREAM_TIMEOUT. complete() uses COMPLETE_TIMEOUT.
# Internal calls (classifier, writeback) should pass a tighter override.

COMPLETE_TIMEOUT = float(os.getenv("LLM_COMPLETE_TIMEOUT", "60"))  # ← Increase
STREAM_TIMEOUT   = float(os.getenv("LLM_STREAM_TIMEOUT",   "60"))  # ← Increase

# How many times to retry on transient failure before raising LLMError.
# Covers: connection resets, 502/503 from proxy, temporary Ollama unavailability.
# Does NOT retry on 4xx (bad request) — that's a code bug, not a transient failure.
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

# Seconds to wait before first retry. Doubles each attempt (1s → 2s).
RETRY_BASE_DELAY = 1.0


# ─── error ────────────────────────────────────────────────────────────────────

class LLMError(Exception):
    """Raised when LiteLLM returns an unrecoverable error or all retries fail."""


# ─── client ───────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Async HTTP client for the LiteLLM proxy.

    Why a class and not bare functions:
      - Owns the httpx connection pool (one pool = one TCP handshake, reused)
      - Holds base_url and tier map as instance state
      - Trivially replaceable in tests with a fake implementation
    """

    def __init__(
        self,
        base_url: str | None = None,
        tier_models: dict[int, str] | None = None,
    ):
        self.base_url    = (base_url or os.getenv("LITELLM_BASE_URL", "http://localhost:4000")).rstrip("/")
        self.tier_models = tier_models or TIER_MODELS

        # One connection pool shared across all requests — never create per-call
        self._client = httpx.AsyncClient(timeout=60.0)

    # ─── public API ───────────────────────────────────────────────────────────

    async def stream(
        self,
        messages: list[dict],
        tier: int = 2,
        timeout: float = STREAM_TIMEOUT,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response token by token.

        Yields each text delta as it arrives from the proxy.
        Use for all user-facing responses — voice, webui, telegram long replies.

        Retries are NOT applied to streaming calls: once the stream starts,
        a retry would duplicate output. Transient failures before the first
        chunk raise LLMError immediately.
        """
        model = self._resolve_model(tier)
        url   = f"{self.base_url}/chat/completions"
        t0 = time.perf_counter()
        logger.debug("LLM stream start  model=%s tier=%d", model, tier)
        trace(
            "LLM.stream start model=%s tier=%d timeout=%.1f messages=%d",
            model,
            tier,
            timeout,
            len(messages),
        )

        try:
            async with self._client.stream(
                "POST",
                url,
                json={
                    "model": model, 
                    "messages": messages, 
                    "stream": True,
                    "max_tokens": timeout * 50 if timeout else None, # heuristic but safer than None
                },
                timeout=timeout,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        trace("LLM.stream done marker received model=%s", model)
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        # Malformed SSE chunk — skip and keep streaming
                        continue

            logger.debug("LLM stream done   model=%s tier=%d duration=%.2fs", model, tier, time.perf_counter() - t0)
            trace("LLM.stream completed model=%s tier=%d elapsed=%.2fs", model, tier, time.perf_counter() - t0)

        except httpx.HTTPStatusError as e:
            trace(
                "LLM.stream http_status_error model=%s tier=%d status=%d elapsed=%.2fs",
                model,
                tier,
                e.response.status_code,
                time.perf_counter() - t0,
            )
            logger.warning("LLM stream error  model=%s tier=%d HTTP %d duration=%.2fs", model, tier, e.response.status_code, time.perf_counter() - t0)
            raise LLMError(
                f"LiteLLM HTTP {e.response.status_code} on stream (tier={tier})"
            ) from e
        except httpx.TimeoutException:
            trace("LLM.stream timeout model=%s tier=%d timeout=%.1f", model, tier, timeout)
            logger.warning("LLM stream timeout  model=%s tier=%d after %.2fs", model, tier, time.perf_counter() - t0)
            raise LLMError(f"Stream timed out after {timeout}s (tier={tier})") from None
        except httpx.RequestError as e:
            trace("LLM.stream request_error model=%s tier=%d error=%s", model, tier, e)
            logger.warning("LLM stream unreachable  model=%s tier=%d error=%s", model, tier, e)
            raise LLMError(f"Could not reach LiteLLM at {self.base_url}: {e}") from e

        
    async def complete(
        self,
        messages: list[dict],
        tier: int = 1,
        timeout: float = COMPLETE_TIMEOUT,
        retries: int = MAX_RETRIES,
    ) -> str:
        """Return the full response as a single string."""
        model = self._resolve_model(tier)
        url   = f"{self.base_url}/chat/completions"
        last_error: Exception | None = None
        t0 = time.perf_counter()
        
        logger.info("LLM complete start  model=%s tier=%d timeout=%.1fs retries=%d", 
                    model, tier, timeout, retries)
        trace("Messages for %s: %s", model, messages)
        
        for attempt in range(retries + 1):
            if attempt > 0:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "LLM complete retry  model=%s attempt=%d/%d delay=%.1fs error=%s",
                    model, attempt + 1, retries + 1, delay, last_error,
                )
                await asyncio.sleep(delay)
            
            try:
                request_payload = {
                    "model": model, 
                    "messages": messages, 
                    "stream": False,
                    "max_tokens": 2048,
                }
                trace("POST to %s with payload keys: %s", url, list(request_payload.keys()))
                
                response = await self._client.post(
                    url,
                    json=request_payload,
                    timeout=timeout,
                )
                
                trace("Response status: %d", response.status_code)
                
                if 400 <= response.status_code < 500:
                    error_detail = response.text[:200] if hasattr(response, 'text') else "unknown"
                    logger.error("4xx error from LiteLLM: %s", error_detail)
                    raise LLMError(
                        f"LiteLLM HTTP {response.status_code} (tier={tier}) — {error_detail}"
                    )
                
                response.raise_for_status()
                resp_json = response.json()
                trace("Response JSON keys: %s", list(resp_json.keys()))
                
                result = resp_json["choices"][0]["message"]["content"]
                
                logger.info(
                    "LLM complete done   model=%s tier=%d duration=%.2fs chars=%d",
                    model, tier, time.perf_counter() - t0, len(result),
                )
                trace("Response text: %s...", result[:200])
                return result
                
            except LLMError:
                raise
            except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError) as e:
                last_error = e
                trace("Exception type: %s, message: %s", type(e).__name__, str(e)[:100])
                continue
            except (KeyError, IndexError) as e:
                raise LLMError(f"Unexpected LiteLLM response shape: {e}") from e
        
        logger.error(
            "LLM complete FAILED  model=%s tier=%d attempts=%d duration=%.2fs error=%s",
            model, tier, retries + 1, time.perf_counter() - t0, last_error,
        )
        raise LLMError(f"LiteLLM failed after {retries + 1} attempts (tier={tier}): {last_error}")

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tier: int = 2,
        timeout: float = COMPLETE_TIMEOUT,
        retries: int = MAX_RETRIES,
    ) -> dict:
        """
        Call the LLM with tool/function definitions and return structured response.

        The LLM can choose to:
          1. Call one or more tools
          2. Return text without calling tools

        Converts OpenAI function calling format to Anthropic-style content blocks
        for compatibility with the orchestrator.

        Args:
            messages: standard OpenAI-format message list.
            tools:    list of tool definitions with name, description, input_schema.
            tier:     which model tier to use (default 2).
            timeout:  per-attempt timeout in seconds.
            retries:  max retry attempts after first failure (0 = no retry).

        Returns:
            Dict with "content" key containing list of blocks:
              - {"type": "text", "text": "..."}
              - {"type": "tool_use", "name": "...", "input": {...}, "id": "..."}
        """
        model = self._resolve_model(tier)
        url   = f"{self.base_url}/chat/completions"
        last_error: Exception | None = None
        t0 = time.perf_counter()
        logger.info("LLM tools start  model=%s tier=%d tool_count=%d", model, tier, len(tools))
        trace(
            "LLM.tools start model=%s tier=%d timeout=%.1f retries=%d input_messages=%d",
            model,
            tier,
            timeout,
            retries,
            len(messages),
        )

        # Convert internal Anthropic-style messages to OpenAI-compatible messages.
        openai_messages = self._to_openai_tool_messages(messages)
        trace("LLM.tools converted_messages=%d", len(openai_messages))

        # Convert tool schemas to OpenAI function format
        functions = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            }
            for tool in tools
        ]

        # Preferred payload for modern OpenAI-compatible tool calling.
        tool_defs = [
            {
                "type": "function",
                "function": fn,
            }
            for fn in functions
        ]

        for attempt in range(retries + 1):
            trace("LLM.tools attempt=%d/%d model=%s", attempt + 1, retries + 1, model)

            if attempt > 0:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "LLM tools retry  model=%s attempt=%d/%d delay=%.1fs error=%s",
                    model, attempt + 1, retries + 1, delay, last_error,
                )
                await asyncio.sleep(delay)

            try:
                # Try modern tools payload first; on 400 fall back to legacy
                # functions payload for older backends.
                request_payload = {
                    "model": model,
                    "messages": openai_messages,
                    "stream": False,
                    "tools": tool_defs,
                    "tool_choice": "auto",
                    "max_tokens": 2048,
                }
                trace("LLM.tools post modern payload_keys=%s", list(request_payload.keys()))
                response = await self._client.post(url, json=request_payload, timeout=timeout)

                if response.status_code == 400:
                    trace("LLM.tools modern 400, trying legacy functions payload")
                    legacy_payload = {
                        "model": model,
                        "messages": openai_messages,
                        "stream": False,
                        "functions": functions,
                        "function_call": "auto",
                        "max_tokens": 2048,
                    }
                    response = await self._client.post(url, json=legacy_payload, timeout=timeout)
                    trace("LLM.tools legacy response_status=%d", response.status_code)

                if 400 <= response.status_code < 500:
                    detail = ""
                    try:
                        detail = response.text[:300]
                    except Exception:
                        detail = ""
                    trace(
                        "LLM.tools client_error status=%d detail_preview=%r",
                        response.status_code,
                        detail[:120],
                    )
                    raise LLMError(
                        f"LiteLLM HTTP {response.status_code} (tier={tier}) — check prompt/auth. {detail}"
                    )

                response.raise_for_status()

                # Convert OpenAI response to Anthropic format
                result = self._convert_to_anthropic_format(
                    response.json()["choices"][0]["message"]
                )
                trace(
                    "LLM.tools converted_result_blocks=%d elapsed=%.2fs",
                    len(result.get("content", [])),
                    time.perf_counter() - t0,
                )
                logger.info(
                    "LLM tools done   model=%s tier=%d duration=%.2fs blocks=%d",
                    model, tier, time.perf_counter() - t0, len(result.get("content", [])),
                )
                return result

            except LLMError:
                raise

            except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError) as e:
                last_error = e
                trace("LLM.tools transient_error type=%s message=%s", type(e).__name__, str(e)[:150])
                continue

            except (KeyError, IndexError) as e:
                raise LLMError(f"Unexpected LiteLLM response shape: {e}") from e

        logger.error(
            "LLM tools FAILED  model=%s tier=%d attempts=%d duration=%.2fs error=%s",
            model, tier, retries + 1, time.perf_counter() - t0, last_error,
        )
        trace(
            "LLM.tools failed model=%s tier=%d attempts=%d error=%s",
            model,
            tier,
            retries + 1,
            last_error,
        )
        raise LLMError(
            f"LiteLLM failed after {retries + 1} attempts (tier={tier}): {last_error}"
        )

    async def aclose(self) -> None:
        """Release the HTTP connection pool. Call this on app shutdown."""
        await self._client.aclose()

    async def warmup(self, tiers: list[int] | None = None) -> None:
        """
        Send a trivial prompt to local models so Ollama loads them into memory.

        Call once during startup. Without this, the first real request pays
        a 5-15s cold-start penalty while Ollama loads model weights.

        Only warms local tiers (1, 2) by default. Cloud tiers (3) are skipped
        — they don't benefit from warm-up.

        Args:
            tiers: which tiers to warm. Defaults to [1, 2] (local Ollama models).
        """
        warm_tiers = tiers or [1, 2]
        warmup_msg = [{"role": "user", "content": "hi"}]

        for tier in warm_tiers:
            model = self._resolve_model(tier)
            t0 = time.perf_counter()
            try:
                await self.complete(warmup_msg, tier=tier, timeout=30.0, retries=0)
                logger.info(
                    "Warmed up model=%s tier=%d (%.2fs)",
                    model, tier, time.perf_counter() - t0,
                )
            except Exception as e:
                # Non-fatal — model will load on first real request instead
                logger.warning(
                    "Warmup failed for model=%s tier=%d (%.2fs): %s",
                    model, tier, time.perf_counter() - t0, e,
                )

    # ─── internal helpers ─────────────────────────────────────────────────────

    def _convert_to_anthropic_format(self, message: dict) -> dict:
        """
        Convert OpenAI message format to Anthropic-style content blocks.

        OpenAI format:
            {"content": "...", "tool_calls": [...]}

        Anthropic format:
            {"content": [{"type": "text", "text": "..."}, {"type": "tool_use", ...}]}
        """
        content = []

        # Add text content if present
        if message.get("content"):
            content.append({
                "type": "text",
                "text": message["content"],
            })

        # Convert function calls to tool_use blocks
        for tool_call in message.get("tool_calls", []):
            # Extract function name and arguments
            func_name = tool_call["function"]["name"]
            args_str = tool_call["function"]["arguments"]
            
            # Parse arguments (they come as JSON string)
            try:
                args = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                args = {}

            content.append({
                "type": "tool_use",
                "id": tool_call["id"],
                "name": func_name,
                "input": args,
            })

        return {"content": content}

    def _to_openai_tool_messages(self, messages: list[dict]) -> list[dict]:
        """
        Convert internal Anthropic-style tool loop messages into OpenAI format.

        Orchestrator stores tool turns like:
            assistant: [{type:text}, {type:tool_use, ...}]
            user:      [{type:tool_result, ...}]

        LiteLLM / OpenAI expects:
            assistant: {content: str|None, tool_calls:[...]}
            tool:      {tool_call_id:..., content:...}
        """
        out: list[dict] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # Normal OpenAI-style messages pass through.
            if isinstance(content, str) or content is None:
                out.append({"role": role, "content": content})
                continue

            # Internal Anthropic-style block list.
            if isinstance(content, list):
                if role == "assistant":
                    text_parts: list[str] = []
                    tool_calls: list[dict] = []

                    for block in content:
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            tool_calls.append({
                                "id": block.get("id", "tool_call"),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input", {})),
                                },
                            })

                    assistant_msg = {
                        "role": "assistant",
                        "content": "".join(text_parts) if text_parts else None,
                    }
                    if tool_calls:
                        assistant_msg["tool_calls"] = tool_calls
                    out.append(assistant_msg)
                    continue

                if role == "user":
                    # user tool_result blocks become tool-role messages.
                    for block in content:
                        if block.get("type") == "tool_result":
                            out.append({
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": block.get("content", ""),
                            })
                    continue

            # Defensive fallback.
            out.append({"role": role, "content": str(content)})

        return out

    def _resolve_model(self, tier: int) -> str:
        """
        Map a tier number to a model name.
        Logs a warning and falls back to tier 2 if an unknown tier is given.
        """
        if tier not in self.tier_models:
            # Unknown tier — don't crash, but don't silently use the wrong model
            import warnings
            warnings.warn(
                f"Unknown tier={tier}, falling back to tier 2. Check your call site.",
                stacklevel=3,
            )
            return self.tier_models[2]
        return self.tier_models[tier]


# ─── singleton ────────────────────────────────────────────────────────────────

# Use this everywhere in the codebase.
# Only create a new LLMClient instance in tests or if you need a different proxy.
llm = LLMClient()