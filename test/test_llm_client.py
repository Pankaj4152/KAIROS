"""
Tests for runtime/llm/client.py

Covers:
  - _resolve_model: tier 1/2/3 → correct model name
  - _resolve_model: unknown tier → warning + fallback to tier 2
  - Constructor defaults (base_url, tier_models)
  - Custom base_url / tier_models override
  - complete_with_tools: converts OpenAI format to Anthropic format
  - _convert_to_anthropic_format: tool_calls → tool_use blocks
"""

import asyncio
import json
import logging
import warnings
from unittest.mock import AsyncMock, MagicMock

import pytest

logger = logging.getLogger(__name__)


class TestResolveModel:
    def test_tier_1(self):
        from llm.client import LLMClient
        client = LLMClient()
        model = client._resolve_model(1)
        logger.info("Tier 1 → model=%r", model)
        assert model == "tier1"

    def test_tier_2(self):
        from llm.client import LLMClient
        client = LLMClient()
        model = client._resolve_model(2)
        logger.info("Tier 2 → model=%r", model)
        assert model == "tier2"

    def test_tier_3(self):
        from llm.client import LLMClient
        client = LLMClient()
        model = client._resolve_model(3)
        logger.info("Tier 3 → model=%r", model)
        assert model == "tier3"

    def test_unknown_tier_warns_and_fallback(self):
        from llm.client import LLMClient
        client = LLMClient()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = client._resolve_model(99)
        logger.info("Unknown tier 99 → model=%r, warnings=%d", model, len(w))
        assert model == "tier2"  # fallback
        assert len(w) == 1
        assert "Unknown tier" in str(w[0].message)


class TestConstructor:
    def test_default_base_url(self):
        from llm.client import LLMClient
        client = LLMClient()
        logger.info("Default base_url: %s", client.base_url)
        assert "localhost" in client.base_url or "4000" in client.base_url

    def test_custom_base_url(self):
        from llm.client import LLMClient
        client = LLMClient(base_url="http://my-proxy:8080")
        logger.info("Custom base_url: %s", client.base_url)
        assert client.base_url == "http://my-proxy:8080"

    def test_trailing_slash_stripped(self):
        from llm.client import LLMClient
        client = LLMClient(base_url="http://my-proxy:8080/")
        logger.info("Stripped URL: %s", client.base_url)
        assert not client.base_url.endswith("/")

    def test_custom_tier_models(self):
        from llm.client import LLMClient
        custom = {1: "phi3", 2: "haiku", 3: "sonnet"}
        client = LLMClient(tier_models=custom)
        logger.info("Custom tier models: %s", client.tier_models)
        assert client._resolve_model(1) == "phi3"
        assert client._resolve_model(3) == "sonnet"


class TestLLMError:
    def test_error_is_exception(self):
        from llm.client import LLMError
        with pytest.raises(LLMError):
            raise LLMError("test error")
        logger.info("LLMError is a proper exception ✓")


class TestConvertToAnthropicFormat:
    def test_text_only_message(self):
        from llm.client import LLMClient
        client = LLMClient()
        message = {"content": "Hello, world!"}
        result = client._convert_to_anthropic_format(message)
        logger.info("Text-only result: %s", result)
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello, world!"

    def test_tool_call_only(self):
        from llm.client import LLMClient
        client = LLMClient()
        message = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query": "latest AI news"}',
                    },
                }
            ],
        }
        result = client._convert_to_anthropic_format(message)
        logger.info("Tool-only result: %s", result)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["input"]["query"] == "latest AI news"

    def test_text_and_tool_calls(self):
        from llm.client import LLMClient
        client = LLMClient()
        message = {
            "content": "I'll create that event for you.",
            "tool_calls": [
                {
                    "id": "call_456",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query": "AI news 2026"}',
                    },
                }
            ],
        }
        result = client._convert_to_anthropic_format(message)
        logger.info("Mixed result: %s", result)
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "tool_use"

    def test_multiple_tool_calls(self):
        from llm.client import LLMClient
        client = LLMClient()
        message = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "tool_a",
                        "arguments": '{"x": 1}',
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "tool_b",
                        "arguments": '{"y": 2}',
                    },
                },
            ],
        }
        result = client._convert_to_anthropic_format(message)
        logger.info("Multiple tools: %s", result)
        assert len(result["content"]) == 2
        assert result["content"][0]["id"] == "call_1"
        assert result["content"][1]["id"] == "call_2"

    def test_malformed_json_arguments(self):
        from llm.client import LLMClient
        client = LLMClient()
        message = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {
                        "name": "some_tool",
                        "arguments": "invalid json {[ ]",
                    },
                }
            ],
        }
        result = client._convert_to_anthropic_format(message)
        logger.info("Malformed args result: %s", result)
        assert result["content"][0]["input"] == {}  # Empty dict fallback


def run(coro):
    """Run async tests."""
    return asyncio.run(coro)


class TestCompleteWithTools:
    def test_complete_with_tools_returns_anthropic_format(self, monkeypatch):
        from llm.client import LLMClient
        
        client = LLMClient()
        
        # Mock the httpx client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "I'll search for that.",
                        "tool_calls": [
                            {
                                "id": "call_789",
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query": "python async guide"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        
        async def mock_post(*args, **kwargs):
            return mock_response
        
        client._client.post = AsyncMock(return_value=mock_response)
        
        messages = [{"role": "user", "content": "Create an event"}]
        tools = [
            {
                "name": "web_search",
                "description": "Search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                },
            }
        ]
        
        result = run(client.complete_with_tools(messages, tools))
        logger.info("complete_with_tools result: %s", result)
        
        assert "content" in result
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "tool_use"
        assert result["content"][1]["name"] == "web_search"

    def test_complete_with_tools_falls_back_to_legacy_functions_on_400(self):
        from llm.client import LLMClient

        client = LLMClient()

        bad_response = MagicMock()
        bad_response.status_code = 400
        bad_response.text = "tools not supported"

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                        "tool_calls": [],
                    }
                }
            ]
        }

        client._client.post = AsyncMock(side_effect=[bad_response, ok_response])

        result = run(
            client.complete_with_tools(
                messages=[{"role": "user", "content": "hi"}],
                tools=[
                    {
                        "name": "web_search",
                        "description": "Search",
                        "input_schema": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    }
                ],
            )
        )

        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "ok"
        assert client._client.post.await_count == 2


class TestToOpenAIToolMessages:
    def test_passes_normal_messages_through(self):
        from llm.client import LLMClient

        client = LLMClient()
        messages = [{"role": "user", "content": "hello"}]
        converted = client._to_openai_tool_messages(messages)

        assert converted == [{"role": "user", "content": "hello"}]

    def test_converts_assistant_tool_use_blocks(self):
        from llm.client import LLMClient

        client = LLMClient()
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "calling tool"},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "web_search",
                        "input": {"query": "news"},
                    },
                ],
            }
        ]

        converted = client._to_openai_tool_messages(messages)

        assert converted[0]["role"] == "assistant"
        assert converted[0]["content"] == "calling tool"
        assert converted[0]["tool_calls"][0]["id"] == "call_1"
        assert converted[0]["tool_calls"][0]["function"]["name"] == "web_search"

    def test_converts_user_tool_result_blocks(self):
        from llm.client import LLMClient

        client = LLMClient()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "Event created",
                    }
                ],
            }
        ]

        converted = client._to_openai_tool_messages(messages)

        assert converted == [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "Event created",
            }
        ]

