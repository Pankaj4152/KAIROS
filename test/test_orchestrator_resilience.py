"""
Resilience-focused tests for orchestrator fallback behavior.

These tests cover stable unit-level guarantees and keep heavier end-to-end
fallback scenarios explicitly marked for follow-up integration coverage.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm.client import LLMError
from orchestrator.orchestrator import Orchestrator


def run(coro):
    return asyncio.run(coro)


async def _collect(async_gen):
    out = []
    async for token in async_gen:
        out.append(token)
    return out


class TestOrchestratorResilience:
    def test_stream_tier_fallback_on_tier3_failure(self):
        """Tier 3 failure should fall back and stream from tier 2."""
        llm = MagicMock()

        async def _stream(messages, tier, timeout):
            if tier == 3:
                raise LLMError("tier3 unavailable")
            for token in ["ok", " from", " tier2"]:
                yield token

        llm.stream = _stream

        orch = Orchestrator(llm_client=llm)
        tokens = run(_collect(orch._stream_with_tier_fallback(
            messages=[{"role": "user", "content": "hello"}],
            tier=3,
        )))

        assert "".join(tokens) == "ok from tier2"

    @pytest.mark.skip(reason="Needs deeper tool-loop integration harness")
    def test_tool_loop_retry_on_tier_failure(self):
        """Reserved for integration test of per-round tool-loop tier fallback."""
        pass

    @pytest.mark.skip(reason="Needs deterministic fixture for full tool-loop rounds")
    def test_max_tool_rounds_degradation(self):
        """Reserved for integration test of max-round degradation path."""
        pass

    def test_session_history_corruption_recovery(self, monkeypatch):
        """Session history read failures should recover with empty history."""
        orch = Orchestrator(llm_client=MagicMock())

        async def _boom(*args, **kwargs):
            raise RuntimeError("corrupted session")

        monkeypatch.setattr("orchestrator.orchestrator.get_history", _boom)

        history = run(orch._safe_get_history("session-1", last_n=8))
        assert history == []
