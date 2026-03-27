"""
Tests for runtime/tools/executor.py + registry.py

Covers:
  - Unknown tool name → error string
  - Missing required field → validation error
  - Extra field (additionalProperties: false) → validation error
  - Valid input → calls handler, returns result
  - Handler exception → error string (no crash)
"""

import asyncio
import logging
from unittest.mock import AsyncMock, patch

import pytest

logger = logging.getLogger(__name__)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestToolExecutor:
    def test_unknown_tool_returns_error(self):
        from tools.executor import execute
        result = run(execute("fly_drone", {"target": "moon"}))
        logger.info("Unknown tool result: %s", result)
        assert "not available" in result.lower() or "error" in result.lower()

    def test_missing_required_field(self):
        from tools.executor import execute
        result = run(execute("web_search", {}))  # missing "query"
        logger.info("Missing field result: %s", result)
        assert "error" in result.lower()

    def test_extra_field_rejected(self):
        from tools.executor import execute
        result = run(execute("web_search", {"query": "test", "evil_param": "hack"}))
        logger.info("Extra field result: %s", result)
        assert "error" in result.lower()

    def test_valid_input_calls_handler(self):
        from tools.executor import execute
        from tools import registry

        mock_handler = AsyncMock(return_value="search results here")
        original = registry.REGISTRY["web_search"]["handler"]
        registry.REGISTRY["web_search"]["handler"] = mock_handler

        try:
            result = run(execute("web_search", {"query": "AI news"}))
            logger.info("Valid call result: %s", result)
            assert "search results here" in result
            mock_handler.assert_called_once_with(query="AI news")
        finally:
            registry.REGISTRY["web_search"]["handler"] = original

    def test_handler_exception_returns_error(self):
        from tools.executor import execute
        from tools import registry

        mock_handler = AsyncMock(side_effect=RuntimeError("API timeout"))
        original = registry.REGISTRY["web_search"]["handler"]
        registry.REGISTRY["web_search"]["handler"] = mock_handler

        try:
            result = run(execute("web_search", {"query": "test"}))
            logger.info("Handler exception result: %s", result)
            assert "error" in result.lower()
            assert "API timeout" in result
        finally:
            registry.REGISTRY["web_search"]["handler"] = original


class TestRegistry:
    def test_web_search_registered(self):
        from tools.registry import REGISTRY
        assert "web_search" in REGISTRY
        logger.info("Registered tools: %s", list(REGISTRY.keys()))

    def test_schema_has_required_query(self):
        from tools.registry import REGISTRY
        schema = REGISTRY["web_search"]["schema"]
        assert "query" in schema["properties"]
        assert "query" in schema["required"]
        logger.info("web_search schema validated ✓")
