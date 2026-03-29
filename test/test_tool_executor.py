"""Tests for runtime/tools/executor.py."""

import asyncio
from unittest.mock import AsyncMock

import pytest


def run(coro):
    """Run async test helpers without requiring pytest-asyncio."""
    return asyncio.run(coro)


def _sample_schema(required=True):
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
        },
        "additionalProperties": False,
    }
    if required:
        schema["required"] = ["query"]
    return schema


def test_unknown_tool_returns_not_available():
    from tools import executor

    result = run(executor.execute("does_not_exist", {"query": "x"}))
    assert "not available" in result.lower()


def test_disabled_tool_rejected(monkeypatch):
    from tools import executor

    fake_registry = {
        "web_search": {
            "enabled": False,
            "schema": _sample_schema(),
            "handler": lambda: (lambda **kwargs: "ok"),
        }
    }
    monkeypatch.setattr(executor, "REGISTRY", fake_registry)
    monkeypatch.setattr(executor, "get_eligibility", lambda: {"web_search": False})

    result = run(executor.execute("web_search", {"query": "x"}))
    assert "disabled" in result.lower()


def test_ineligible_tool_rejected(monkeypatch):
    from tools import executor

    fake_registry = {
        "web_search": {
            "enabled": True,
            "schema": _sample_schema(),
            "handler": lambda: (lambda **kwargs: "ok"),
        }
    }
    monkeypatch.setattr(executor, "REGISTRY", fake_registry)
    monkeypatch.setattr(executor, "get_eligibility", lambda: {"web_search": False})

    result = run(executor.execute("web_search", {"query": "x"}))
    assert "not configured" in result.lower()


def test_input_validation_failure(monkeypatch):
    from tools import executor

    fake_registry = {
        "web_search": {
            "enabled": True,
            "schema": _sample_schema(required=True),
            "handler": lambda: (lambda **kwargs: "ok"),
        }
    }
    monkeypatch.setattr(executor, "REGISTRY", fake_registry)
    monkeypatch.setattr(executor, "get_eligibility", lambda: {"web_search": True})

    result = run(executor.execute("web_search", {}))
    assert "invalid input" in result.lower()


def test_loader_failure_returns_error(monkeypatch):
    from tools import executor

    def broken_loader():
        raise RuntimeError("import failure")

    fake_registry = {
        "web_search": {
            "enabled": True,
            "schema": _sample_schema(),
            "handler": broken_loader,
        }
    }
    monkeypatch.setattr(executor, "REGISTRY", fake_registry)
    monkeypatch.setattr(executor, "get_eligibility", lambda: {"web_search": True})

    result = run(executor.execute("web_search", {"query": "x"}))
    assert "could not load tool" in result.lower()


def test_async_handler_success(monkeypatch):
    from tools import executor

    handler = AsyncMock(return_value="results")
    fake_registry = {
        "web_search": {
            "enabled": True,
            "schema": _sample_schema(),
            "handler": lambda: handler,
        }
    }
    monkeypatch.setattr(executor, "REGISTRY", fake_registry)
    monkeypatch.setattr(executor, "get_eligibility", lambda: {"web_search": True})

    result = run(executor.execute("web_search", {"query": "ai"}))
    assert result == "results"
    handler.assert_awaited_once_with(query="ai")


def test_sync_handler_success(monkeypatch):
    from tools import executor

    def sync_handler(query):
        return f"sync:{query}"

    fake_registry = {
        "web_search": {
            "enabled": True,
            "schema": _sample_schema(),
            "handler": lambda: sync_handler,
        }
    }
    monkeypatch.setattr(executor, "REGISTRY", fake_registry)
    monkeypatch.setattr(executor, "get_eligibility", lambda: {"web_search": True})

    result = run(executor.execute("web_search", {"query": "ai"}))
    assert result == "sync:ai"


def test_timeout_returns_error(monkeypatch):
    from tools import executor

    async def slow_handler(query):
        await asyncio.sleep(0.05)
        return query

    fake_registry = {
        "web_search": {
            "enabled": True,
            "schema": _sample_schema(),
            "handler": lambda: slow_handler,
        }
    }
    monkeypatch.setattr(executor, "REGISTRY", fake_registry)
    monkeypatch.setattr(executor, "get_eligibility", lambda: {"web_search": True})
    monkeypatch.setattr(executor, "TOOL_TIMEOUT_SECONDS", 0.01)

    result = run(executor.execute("web_search", {"query": "ai"}))
    assert "timed out" in result.lower()


def test_handler_exception_returns_error(monkeypatch):
    from tools import executor

    async def bad_handler(query):
        raise RuntimeError("boom")

    fake_registry = {
        "web_search": {
            "enabled": True,
            "schema": _sample_schema(),
            "handler": lambda: bad_handler,
        }
    }
    monkeypatch.setattr(executor, "REGISTRY", fake_registry)
    monkeypatch.setattr(executor, "get_eligibility", lambda: {"web_search": True})

    result = run(executor.execute("web_search", {"query": "ai"}))
    assert "error running" in result.lower()
    assert "boom" in result.lower()
