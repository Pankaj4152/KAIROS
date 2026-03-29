"""
Tests for runtime/memory/session_store.py

Covers:
  - append_turn creates session file with correct structure
  - get_history returns last N turns in OpenAI format
  - get_history prepends summary after compaction
  - compact keeps last 2 turns + stores summary
  - compact rejects empty summary
  - needs_compaction triggers at MAX_TURNS
"""

import asyncio
import json
import logging
import os

import pytest

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _patch_sessions_dir(tmp_data_dir, monkeypatch):
    """Point session_store at a temp directory."""
    sessions_dir = os.path.join(tmp_data_dir, "sessions")
    monkeypatch.setattr("memory.session_store.DATA_DIR", tmp_data_dir)
    monkeypatch.setattr("memory.session_store.SESSIONS_DIR", sessions_dir)
    logger.info("Sessions dir: %s", sessions_dir)


def run(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.run(coro)


# ── tests ─────────────────────────────────────────────────────────────────────

class TestAppendTurn:
    def test_creates_session_file(self, tmp_data_dir):
        from memory.session_store import append_turn, _session_path
        run(append_turn("sess-001", "user", "Hello Kairos"))

        path = _session_path("sess-001")
        assert os.path.exists(path), f"Session file not created at {path}"

        with open(path, "r") as f:
            data = json.load(f)
        logger.info("Session data: %s", json.dumps(data, indent=2)[:200])
        assert data["session_id"] == "sess-001"
        assert len(data["turns"]) == 1
        assert data["turns"][0]["role"] == "user"
        assert data["turns"][0]["content"] == "Hello Kairos"

    def test_appends_multiple_turns(self):
        from memory.session_store import append_turn, get_session
        run(append_turn("sess-002", "user", "What tasks do I have?"))
        run(append_turn("sess-002", "assistant", "You have 3 open tasks."))

        session = run(get_session("sess-002"))
        logger.info("Turn count: %d", len(session["turns"]))
        assert len(session["turns"]) == 2
        assert session["turns"][0]["role"] == "user"
        assert session["turns"][1]["role"] == "assistant"

    def test_stores_metadata(self):
        from memory.session_store import append_turn, get_session
        run(append_turn("sess-003", "user", "Hi", meta={"channel": "voice"}))

        session = run(get_session("sess-003"))
        turn = session["turns"][0]
        logger.info("Turn meta: %s", turn)
        assert turn["channel"] == "voice"


class TestGetHistory:
    def test_returns_last_n_turns(self):
        from memory.session_store import append_turn, get_history
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            run(append_turn("sess-hist", role, f"Message {i}"))

        history = run(get_history("sess-hist", last_n=4))
        logger.info("History (last 4): %s", [m["content"] for m in history])
        assert len(history) == 4
        assert history[0]["content"] == "Message 6"

    def test_returns_openai_format(self):
        from memory.session_store import append_turn, get_history
        run(append_turn("sess-fmt", "user", "Hello"))
        run(append_turn("sess-fmt", "assistant", "Hi there"))

        history = run(get_history("sess-fmt", last_n=8))
        for msg in history:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ("user", "assistant", "system")
        logger.info("OpenAI format validated ✓")

    def test_prepends_summary_after_compaction(self):
        from memory.session_store import append_turn, compact, get_history
        for i in range(5):
            run(append_turn("sess-summary", "user", f"Q{i}"))
            run(append_turn("sess-summary", "assistant", f"A{i}"))

        run(compact("sess-summary", "User asked 5 questions about various topics."))

        history = run(get_history("sess-summary", last_n=8))
        logger.info("History after compaction: %s", [m["role"] for m in history])
        assert history[0]["role"] == "system"
        assert "summary" in history[0]["content"].lower()


class TestCompaction:
    def test_keeps_last_two_turns(self):
        from memory.session_store import append_turn, compact, get_session
        for i in range(10):
            run(append_turn("sess-compact", "user", f"Msg {i}"))

        run(compact("sess-compact", "Summary of first 8 messages."))

        session = run(get_session("sess-compact"))
        logger.info("Turns after compaction: %d (expected 2)", len(session["turns"]))
        assert len(session["turns"]) == 2
        assert session["summary"] == "Summary of first 8 messages."

    def test_rejects_empty_summary(self):
        from memory.session_store import append_turn, compact
        run(append_turn("sess-empty", "user", "Hello"))

        with pytest.raises(ValueError, match="must not be empty"):
            run(compact("sess-empty", ""))
        logger.info("Empty summary correctly rejected")

    def test_rejects_whitespace_summary(self):
        from memory.session_store import append_turn, compact
        run(append_turn("sess-ws", "user", "Hello"))

        with pytest.raises(ValueError, match="must not be empty"):
            run(compact("sess-ws", "   "))
        logger.info("Whitespace-only summary correctly rejected")


class TestNeedsCompaction:
    def test_triggers_at_threshold(self, monkeypatch):
        from memory.session_store import append_turn, needs_compaction
        monkeypatch.setattr("memory.session_store.MAX_TURNS", 5)

        for i in range(6):
            run(append_turn("sess-threshold", "user", f"Turn {i}"))

        result = run(needs_compaction("sess-threshold"))
        logger.info("needs_compaction after 6 turns (max=5): %s", result)
        assert result is True

    def test_does_not_trigger_below_threshold(self, monkeypatch):
        from memory.session_store import append_turn, needs_compaction
        monkeypatch.setattr("memory.session_store.MAX_TURNS", 20)

        for i in range(3):
            run(append_turn("sess-below", "user", f"Turn {i}"))

        result = run(needs_compaction("sess-below"))
        logger.info("needs_compaction after 3 turns (max=20): %s", result)
        assert result is False
