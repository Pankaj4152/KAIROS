"""
Gateway — Session resolver (gateway/session.py)

Purpose:
    Track which conversations are currently alive and hand back
    the correct session_id for every incoming event.

    This is a LIVENESS registry only — it answers:
      "Is this session still active, or should we start fresh?"

    It does NOT store conversation history or message content.
    That is the job of memory/session_store.py.

                  ┌─────────────────────────────────┐
    KairosEvent   │  session.py                     │   session_id
    ──────────────►  resolve_session(incoming_id)   ├──────────────► orchestrator
                  │  _sessions: {id → {timestamps}} │
                  └─────────────────────────────────┘

Boundary rules:
    - Reads from: nothing (pure in-memory state)
    - Writes to:  _sessions dict (in-process only)
    - Does NOT touch: SQLite, JSON files, or any I/O

    In-memory dict is intentional for MVP (single user, single process).
    Sessions are re-created on restart — channel adapters (Telegram chat_id,
    WebUI tab UUID) provide natural re-entry points.

What this file does NOT do:
    - No authentication (Telegram auth is in channels/telegram.py)
    - No conversation history (that is memory/session_store.py)
    - No business logic
"""

import time
import uuid
from typing import Optional
from dataclasses import dataclass


# All active sessions keyed by session_id.
# Plain dict is fine — single user, single process.
_sessions: dict[str, dict] = {}

from config.settings import SESSION_TIMEOUT, PURGE_PROBABILITY

# ─── session record ───────────────────────────────────────────────────────────

@dataclass
class SessionMeta:
    """Everything we know about an active session."""
    session_id: str
    user_id:    str
    channel:    str       # "voice" | "telegram" | "webui" | "cron"
    created_at: float     # unix timestamp
    last_active: float    # unix timestamp — updated on every resolve

def resolve_session(
    channel: str,
    user_id: str,
    incoming_session_id: str | None = None,
) -> str:
    """
    Resume a warm session or create a new one.

    Args:
        channel:             originating channel — stored in metadata.
        user_id:             who is sending — stored in metadata.
        incoming_session_id: hint from the channel layer.
                             Telegram passes chat_id. WebUI passes a tab UUID.
                             Cron passes None → always gets a fresh session.

    Returns:
        session_id to use for this request. Caller passes this to
        session_store.py to load actual conversation history.

    Session ID rules:
        - Cron: always a fresh UUID (cron jobs don't share context)
        - Existing warm session: resume it, bump last_active
        - Expired session: start fresh — new UUID, not the old channel id
        - No session provided: generate a new UUID
    """
    import random

    # Cron jobs never share context — always isolated
    if channel == "cron":
        return _create_session(channel, user_id, session_id=f"cron-{uuid.uuid4()}")

    now = time.time()
    expired_incoming = False

    # Occasional cleanup — keeps memory from growing unbounded
    if random.random() < PURGE_PROBABILITY:
        _purge_expired()

    # Try to resume an existing warm session
    if incoming_session_id and incoming_session_id in _sessions:
        meta = _sessions[incoming_session_id]

        if now - meta.last_active < SESSION_TIMEOUT:
            meta.last_active = now
            return incoming_session_id

        # Expired — remove it. Fall through to create a fresh one.
        # NOTE: we do NOT reuse incoming_session_id here. A Telegram chat_id
        # that expired gets a new UUID — clean context window, same user memory.
        del _sessions[incoming_session_id]
        expired_incoming = True

    # First message for a channel-provided session hint (web tab id, chat id).
    # Store it as-is so the next message with the same hint can resume.
    if incoming_session_id and not expired_incoming:
        return _create_session(channel, user_id, session_id=incoming_session_id)

    return _create_session(channel, user_id)


def get_active_sessions() -> list[SessionMeta]:
    """
    All currently warm sessions. For debugging and diagnostics only.
    Not in the hot path.
    """
    now = time.time()
    return [
        meta for meta in _sessions.values()
        if now - meta.last_active < SESSION_TIMEOUT
    ]


# ─── internal helpers ─────────────────────────────────────────────────────────

def _create_session(channel: str, user_id: str, session_id: str | None = None) -> str:
    """
    Register a new session in the liveness cache and return its ID.
    Always generates a fresh UUID unless session_id is explicitly provided
    (only cron does this, to embed the job context in the ID).
    """
    now = time.time()
    sid = session_id or str(uuid.uuid4())

    _sessions[sid] = SessionMeta(
        session_id=sid,
        user_id=user_id,
        channel=channel,
        created_at=now,
        last_active=now,
    )

    return sid


def _purge_expired() -> int:
    """
    Remove expired sessions from the liveness cache.
    Called probabilistically from resolve_session — never call manually
    except in tests or on startup.
    Returns the number of sessions removed.
    """
    now = time.time()
    expired = [
        sid for sid, meta in _sessions.items()
        if now - meta.last_active >= SESSION_TIMEOUT
    ]
    for sid in expired:
        del _sessions[sid]
    return len(expired)