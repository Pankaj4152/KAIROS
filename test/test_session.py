"""
Tests for runtime/gateway/session.py

Covers:
  - resolve_session creates new session, returns UUID
  - Re-resolve before timeout → same session_id
  - Re-resolve after timeout → new session_id
  - Cron always gets fresh session
  - _purge_expired removes stale sessions
  - get_active_sessions only returns warm sessions
"""

import logging
import time

import pytest

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _clear_sessions():
    """Clear the in-memory sessions dict before each test."""
    from gateway.session import _sessions
    _sessions.clear()
    yield
    _sessions.clear()


# ── resolve ───────────────────────────────────────────────────────────────────

class TestResolveSession:
    def test_creates_new_session(self):
        from gateway.session import resolve_session, _sessions
        sid = resolve_session("webui", "local")
        logger.info("New session: %s", sid)
        assert sid in _sessions
        assert _sessions[sid].channel == "webui"
        assert _sessions[sid].user_id == "local"

    def test_resumes_warm_session(self):
        from gateway.session import resolve_session
        sid1 = resolve_session("webui", "local", incoming_session_id="tab-123")
        sid2 = resolve_session("webui", "local", incoming_session_id="tab-123")
        logger.info("First: %s, Second: %s (should match)", sid1, sid2)
        assert sid1 == sid2

    def test_expired_session_gets_new_id(self, monkeypatch):
        from gateway.session import resolve_session, _sessions
        # Create a session and then make it appear expired
        sid1 = resolve_session("webui", "local", incoming_session_id="old-tab")
        _sessions[sid1].last_active = time.time() - 7200  # 2 hours ago

        sid2 = resolve_session("webui", "local", incoming_session_id=sid1)
        logger.info("Expired: %s → New: %s", sid1, sid2)
        assert sid1 != sid2

    def test_cron_always_fresh(self):
        from gateway.session import resolve_session
        sid1 = resolve_session("cron", "local")
        sid2 = resolve_session("cron", "local")
        logger.info("Cron sessions: %s, %s (should differ)", sid1, sid2)
        assert sid1 != sid2
        assert sid1.startswith("cron-")
        assert sid2.startswith("cron-")

    def test_no_session_id_generates_uuid(self):
        from gateway.session import resolve_session
        sid = resolve_session("telegram", "user-42")
        logger.info("Generated UUID: %s", sid)
        assert len(sid) == 36  # standard UUID format


# ── purge ─────────────────────────────────────────────────────────────────────

class TestPurgeExpired:
    def test_removes_stale_sessions(self):
        from gateway.session import resolve_session, _purge_expired, _sessions
        sid = resolve_session("webui", "local")
        _sessions[sid].last_active = time.time() - 7200  # expired

        removed = _purge_expired()
        logger.info("Purged %d session(s)", removed)
        assert removed == 1
        assert sid not in _sessions

    def test_keeps_warm_sessions(self):
        from gateway.session import resolve_session, _purge_expired, _sessions
        sid = resolve_session("webui", "local")
        # session is fresh, should survive purge
        removed = _purge_expired()
        logger.info("Purged %d (expected 0), session %s alive: %s", removed, sid, sid in _sessions)
        assert removed == 0
        assert sid in _sessions


# ── get_active_sessions ───────────────────────────────────────────────────────

class TestGetActiveSessions:
    def test_returns_only_warm(self):
        from gateway.session import resolve_session, get_active_sessions, _sessions
        sid1 = resolve_session("webui", "local")
        sid2 = resolve_session("telegram", "user-1")
        _sessions[sid1].last_active = time.time() - 7200  # expired

        active = get_active_sessions()
        active_ids = [s.session_id for s in active]
        logger.info("Active sessions: %s (expected only %s)", active_ids, sid2)
        assert sid2 in active_ids
        assert sid1 not in active_ids
