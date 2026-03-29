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
import os
from typing import AsyncGenerator

import httpx
from dotenv import load_dotenv

load_dotenv()


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
STREAM_TIMEOUT   = float(os.getenv("LLM_STREAM_TIMEOUT",   "30"))
COMPLETE_TIMEOUT = float(os.getenv("LLM_COMPLETE_TIMEOUT", "15"))

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

        try:
            async with self._client.stream(
                "POST",
                url,
                json={"model": model, "messages": messages, "stream": True},
                timeout=timeout,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        # Malformed SSE chunk — skip and keep streaming
                        continue

        except httpx.HTTPStatusError as e:
            raise LLMError(
                f"LiteLLM HTTP {e.response.status_code} on stream (tier={tier})"
            ) from e
        except httpx.TimeoutException:
            raise LLMError(f"Stream timed out after {timeout}s (tier={tier})") from None
        except httpx.RequestError as e:
            raise LLMError(f"Could not reach LiteLLM at {self.base_url}: {e}") from e

    async def complete(
        self,
        messages: list[dict],
        tier: int = 1,
        timeout: float = COMPLETE_TIMEOUT,
        retries: int = MAX_RETRIES,
    ) -> str:
        """
        Return the full response as a single string.

        Use for internal calls: classifier, fact extractor, session compaction.
        Retries automatically on transient failures (connection errors, 5xx).
        Will NOT retry on 4xx — that's a prompt/auth bug.

        Args:
            messages: standard OpenAI-format message list.
            tier:     which model tier to use (default 1 = local phi-3-mini).
            timeout:  per-attempt timeout in seconds.
            retries:  max retry attempts after first failure (0 = no retry).
        """
        model = self._resolve_model(tier)
        url   = f"{self.base_url}/chat/completions"
        last_error: Exception | None = None

        for attempt in range(retries + 1):

            if attempt > 0:
                # Exponential backoff: 1s, 2s, 4s, ...
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                await asyncio.sleep(delay)

            try:
                response = await self._client.post(
                    url,
                    json={"model": model, "messages": messages, "stream": False},
                    timeout=timeout,
                )

                # 4xx = bad request — our bug, not transient. Raise immediately.
                if 400 <= response.status_code < 500:
                    raise LLMError(
                        f"LiteLLM HTTP {response.status_code} (tier={tier}) — check prompt/auth"
                    )

                response.raise_for_status()   # catches 5xx
                return response.json()["choices"][0]["message"]["content"]

            except LLMError:
                raise   # 4xx errors — don't retry

            except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError) as e:
                last_error = e
                continue  # retry

            except (KeyError, IndexError) as e:
                raise LLMError(f"Unexpected LiteLLM response shape: {e}") from e

        raise LLMError(
            f"LiteLLM failed after {retries + 1} attempts (tier={tier}): {last_error}"
        )

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
                }
                response = await self._client.post(url, json=request_payload, timeout=timeout)

                if response.status_code == 400:
                    legacy_payload = {
                        "model": model,
                        "messages": openai_messages,
                        "stream": False,
                        "functions": functions,
                        "function_call": "auto",
                    }
                    response = await self._client.post(url, json=legacy_payload, timeout=timeout)

                if 400 <= response.status_code < 500:
                    detail = ""
                    try:
                        detail = response.text[:300]
                    except Exception:
                        detail = ""
                    raise LLMError(
                        f"LiteLLM HTTP {response.status_code} (tier={tier}) — check prompt/auth. {detail}"
                    )

                response.raise_for_status()
                
                # Convert OpenAI response to Anthropic format
                return self._convert_to_anthropic_format(
                    response.json()["choices"][0]["message"]
                )

            except LLMError:
                raise

            except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError) as e:
                last_error = e
                continue

            except (KeyError, IndexError) as e:
                raise LLMError(f"Unexpected LiteLLM response shape: {e}") from e

        raise LLMError(
            f"LiteLLM failed after {retries + 1} attempts (tier={tier}): {last_error}"
        )

    async def aclose(self) -> None:
        """Release the HTTP connection pool. Call this on app shutdown."""
        await self._client.aclose()

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