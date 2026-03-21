"""
Session store — reads and writes per-session conversation history.

Each session is stored as a JSON file: data/sessions/{session_id}.json

This file answers: "what was said in this session?"
session.py answers:  "is this session currently active?"
These two files are separate concerns — never import one from the other.

File format:
    {
        "session_id":  "uuid",
        "created_at":  "2026-03-21T07:00:00+00:00",
        "last_active": "2026-03-21T09:30:00+00:00",
        "turns": [
            {"role": "user",      "content": "...", "timestamp": "...", "channel": "voice"},
            {"role": "assistant", "content": "...", "timestamp": "...", "tier_used": 2}
        ],
        "summary": null   <- populated after compaction, null otherwise
    }

Compaction:
    When turns exceed MAX_TURNS, writeback.py calls compact() with a summary
    string produced by the tier-1 model. The last 2 turns are preserved so
    the assistant always has some immediate context, even post-compaction.

All public functions are async — file I/O runs in asyncio.to_thread().
"""

import asyncio
import json
import os
from datetime import datetime, timezone

DATA_DIR     = os.getenv("DATA_DIR", "./data")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
MAX_TURNS    = int(os.getenv("SESSION_MAX_TURNS", "20"))

# How many turns to keep after compaction.
# Gives the LLM immediate context even when the rest is summarised.
POST_COMPACT_KEEP = 2


# ─── internal helpers (sync — called inside asyncio.to_thread) ───────────────

def _session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def _now_iso() -> str:
    """Single time source for all timestamps in this file."""
    return datetime.now(timezone.utc).isoformat()


def _load_sync(session_id: str) -> dict:
    """
    Load a session from disk. Returns a fresh empty session if file doesn't exist.
    Sync — always call inside asyncio.to_thread().
    """
    path = _session_path(session_id)
    if not os.path.exists(path):
        now = _now_iso()
        return {
            "session_id":  session_id,
            "created_at":  now,
            "last_active": now,
            "turns":       [],
            "summary":     None,
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_sync(session: dict) -> None:
    """
    Write a session to disk. Stamps last_active before writing.
    Sync — always call inside asyncio.to_thread().
    """
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    session["last_active"] = _now_iso()
    path = _session_path(session["session_id"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)


# ─── public API ───────────────────────────────────────────────────────────────

async def append_turn(
    session_id: str,
    role: str,
    content: str,
    meta: dict | None = None,
) -> None:
    """
    Add one turn to the session file.

    Args:
        session_id: which session to write to.
        role:       "user" or "assistant".
        content:    the message text.
        meta:       optional extra fields stored on the turn — e.g.
                    {"channel": "voice"} for user turns,
                    {"tier_used": 2} for assistant turns.

    Called by writeback.py after every completed response.
    """
    def _run():
        session = _load_sync(session_id)
        turn = {
            "role":      role,
            "content":   content,
            "timestamp": _now_iso(),
        }
        if meta:
            turn.update(meta)
        session["turns"].append(turn)
        _save_sync(session)

    await asyncio.to_thread(_run)


async def get_history(session_id: str, last_n: int = 8) -> list[dict]:
    """
    Return the last N turns as OpenAI-format messages.
    Pass the result directly into the messages list for LLM calls.

    If the session has a summary (post-compaction), it is prepended as a
    system message so the LLM has full context even after turns were dropped.
    """
    def _run():
        session = _load_sync(session_id)
        turns   = session["turns"][-last_n:]
        messages = [{"role": t["role"], "content": t["content"]} for t in turns]

        # Prepend summary as system context if compaction has run
        if session.get("summary"):
            messages.insert(0, {
                "role":    "system",
                "content": f"[Earlier conversation summary]\n{session['summary']}",
            })

        return messages

    return await asyncio.to_thread(_run)


async def get_session(session_id: str) -> dict:
    """
    Return the full session dict.
    Use when you need multiple fields (e.g. summary + turn count) in one read.
    Avoids loading the file twice for two separate calls.
    """
    return await asyncio.to_thread(_load_sync, session_id)


async def needs_compaction(session_id: str) -> bool:
    """True if the session has exceeded MAX_TURNS and should be compacted."""
    session = await get_session(session_id)
    return len(session["turns"]) > MAX_TURNS


async def compact(session_id: str, summary: str) -> None:
    """
    Replace most turns with a summary string produced by the tier-1 model.
    Keeps the last POST_COMPACT_KEEP turns so there's always immediate context.

    Args:
        session_id: which session to compact.
        summary:    paragraph summary from the LLM. Must be non-empty.

    Called by writeback.py when needs_compaction() returns True.
    """
    if not summary or not summary.strip():
        # Refuse to compact with an empty summary — would silently destroy history
        raise ValueError(f"Compaction summary for {session_id!r} must not be empty")

    def _run():
        session = _load_sync(session_id)
        # Keep the most recent turns so context isn't completely cold post-compaction
        session["turns"]   = session["turns"][-POST_COMPACT_KEEP:]
        session["summary"] = summary.strip()
        _save_sync(session)

    await asyncio.to_thread(_run)


async def get_summary(session_id: str) -> str | None:
    """Return the compaction summary for this session, or None if not yet compacted."""
    session = await get_session(session_id)
    return session.get("summary")