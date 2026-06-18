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

COMPLETE_TIMEOUT = float(os.getenv("LLM_COMPLETE_TIMEOUT", "60"))
STREAM_TIMEOUT   = float(os.getenv("LLM_STREAM_TIMEOUT",   "60"))

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
    Async HTTP client for the LiteLLM proxy with robust resilience handling.
    Reuses an underlying connection pool to keep connection handshakes to a minimum.
    """

    def __init__(
        self,
        base_url: str | None = None,
        tier_models: dict[int, str] | None = None,
    ):
        self.base_url    = (base_url or os.getenv("LITELLM_BASE_URL", "http://localhost:4000")).rstrip("/")
        self.tier_models = tier_models or TIER_MODELS

        self.master_key  = os.getenv("GATEWAY_MASTER_KEY", "")

        # 2. Build the correct Bearer token dictionary
        headers = {}
        if self.master_key:
            headers["Authorization"] = f"Bearer {self.master_key}"

        # Set explicitly configured limits for the connection pool
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)    
        # One connection pool shared across all requests — never create per-call
        self._client = httpx.AsyncClient(headers=headers, limits=limits, timeout=60.0)

    # ─── public API ───────────────────────────────────────────────────────────

    async def stream(
        self,
        messages: list[dict],
        tier: int = 2,
        timeout: float = STREAM_TIMEOUT,
        trace_id: str | None = None,  # 1. Accept the trace_id here
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response token by token.
        Yields each text delta as it arrives from the proxy.
        """
        model = self._resolve_model(tier)
        url   = f"{self.base_url}/chat/completions"
        t0 = time.perf_counter()

        logger.debug("LLM stream start  model=%s tier=%d", model, tier)

        # 2. Build the standard OpenAI body payload
        payload = {
            "model": model, 
            "messages": messages, 
            "stream": True,
        }

        # 3. Inject metadata if a trace_id is supplied
        if trace_id:
            payload["metadata"] = {"langfuse_trace_id": trace_id}

        try:
            # Using context manager approach ensures clean connection closing on stream interruptions
            async with self._client.stream("POST", url, json=payload, timeout=timeout) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise httpx.HTTPStatusError(
                        f"Status {response.status_code}", 
                        request=response.request, 
                        response=response
                    )
                
                has_chunks = False
                # response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    has_chunks = True
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue # Skip corrupted or intermediate tracking frames
                
                if not has_chunks:
                    raise LLMError("Stream connected successfully but returned no chunks.")

            logger.debug("LLM stream completed smoothly in %.2fs", time.perf_counter() - t0)

        except httpx.HTTPStatusError as e:
            logger.warning("LLM stream HTTP status error %d", e.response.status_code)
            raise LLMError(f"LiteLLM stream status error: {e.response.status_code}") from e
        except httpx.TimeoutException:
            logger.warning("LLM stream hit a connection timeout ceiling after %.2fs", timeout)
            raise LLMError(f"Stream timed out after {timeout}s") from None
        except httpx.RequestError as e:
            logger.error("LLM stream proxy network connection error: %s", e)
            raise LLMError(f"Could not reach LiteLLM router: {e}") from e
            

        
    async def complete(
        self,
        messages: list[dict],
        tier: int = 1,
        timeout: float = COMPLETE_TIMEOUT,
        retries: int = MAX_RETRIES,
        trace_id: str | None = None,
    ) -> str:
        """Return the full response as a single string."""
        model = self._resolve_model(tier)
        url   = f"{self.base_url}/chat/completions"
        last_error: Exception | None = None
        t0 = time.perf_counter()
        
        
        for attempt in range(retries + 1):
            if attempt > 0:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                # logger.warning(
                #     "LLM complete retry  model=%s attempt=%d/%d delay=%.1fs error=%s",
                #     model, attempt + 1, retries + 1, delay, last_error,
                # )
                logger.warning("LLM retry active. Attempt %d/%d. Waiting %.1fs...", attempt + 1, retries + 1, delay)
                await asyncio.sleep(delay)
            
            try:
                request_payload = {
                    "model": model, 
                    "messages": messages, 
                    "stream": False,
                    # "max_tokens": 2048,
                }
                
                if trace_id:
                    request_payload["metadata"] = {"langfuse_trace_id": trace_id}
                
                response = await self._client.post(
                    url,
                    json=request_payload,
                    timeout=timeout,
                )
                
                
                # CRITICAL FIX: Only immediately abort if it's a structural or credential bug
                if response.status_code in (400, 401, 403):
                    # error_detail = response.text[:200] if hasattr(response, 'text') else "unknown"
                    error_detail = response.text[:200]
                    # logger.error("4xx error from LiteLLM: %s", error_detail)
                    raise LLMError(f"Fatal Client Request Error {response.status_code}: {error_detail}")
                
                # Let 429 (Rate Limits) or 5xx server issues pass through to the retry loop!
                response.raise_for_status()
                resp_json = response.json()
                
                result = resp_json["choices"][0]["message"]["content"]
                
                logger.info("LLM complete done model=%s execution_time=%.2fs", model, time.perf_counter() - t0)
                return result
                
            except LLMError:
                raise # Bubble up fatal bugs instantly
            except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError) as e:
                last_error = e
                logger.debug("Transient failure encountered during execution sequence: %s", e)
                continue
            except (KeyError, IndexError) as e:
                raise LLMError(f"Unexpected response payload shape from proxy endpoint: {e}") from e
        
        
        raise LLMError(f"LiteLLM failed completely after {retries + 1} attempts. Last error: {last_error}")

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tier: int = 2,
        timeout: float = COMPLETE_TIMEOUT,
        retries: int = MAX_RETRIES,
        metadata: dict | None = None,
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



        # Convert internal Anthropic-style messages to OpenAI-compatible messages.
        openai_messages = self._to_openai_tool_messages(messages)

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
 
            if attempt > 0:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
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
                    # "max_tokens": 2048,
                }
                if metadata:
                    request_payload["metadata"] = metadata
                response = await self._client.post(url, json=request_payload, timeout=timeout)

                if response.status_code == 400 and "tools" in response.text:
                    legacy_payload = {
                        "model": model,
                        "messages": openai_messages,
                        "stream": False,
                        "functions": functions,
                        "function_call": "auto",
                        # "max_tokens": 2048,
                    }
                    if metadata:
                        legacy_payload["metadata"] = metadata
                    response = await self._client.post(url, json=legacy_payload, timeout=timeout)

                if response.status_code in (400, 401, 403):
                    raise LLMError(f"Fatal Tool Route Client Error {response.status_code}: {response.text[:200]}")

                response.raise_for_status()

                # Convert OpenAI response to Anthropic format
                result = self._convert_to_anthropic_format(
                    response.json()["choices"][0]["message"]
                )

                logger.info("LLM tools completed model=%s execution_time=%.2fs", model, time.perf_counter() - t0)
                return result

            except LLMError:
                raise

            except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError) as e:
                last_error = e
                continue

            except (KeyError, IndexError) as e:
                raise LLMError(f"Unexpected response payload shape from tool endpoint: {e}") from e

        raise LLMError(f"LiteLLM tool tracking route failed structural runs. Last error: {last_error}")

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
            # model = self._resolve_model(tier)
            # t0 = time.perf_counter()
            try:
                await self.complete(warmup_msg, tier=tier, timeout=15.0, retries=0)
                # logger.info(
                #     "Warmed up model=%s tier=%d (%.2fs)",
                #     model, tier, time.perf_counter() - t0,
                # )
            except Exception as e:
                # Non-fatal — model will load on first real request instead
                # logger.warning(
                #     "Warmup failed for model=%s tier=%d (%.2fs): %s",
                #     model, tier, time.perf_counter() - t0, e,
                # )
                logger.debug("Warmup skipped or unavailable for target tier configuration: %s", e)

    # ─── internal conversion helpers ─────────────────────────────────────────────────────

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
            content.append({"type": "text", "text": message["content"]})

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
            return self.tier_models[2]
        return self.tier_models[tier]


# ─── singleton ────────────────────────────────────────────────────────────────

# Use this everywhere in the codebase.
# Only create a new LLMClient instance in tests or if you need a different proxy.
llm = LLMClient()