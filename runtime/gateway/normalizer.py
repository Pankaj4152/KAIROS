"""
Normalizer Layer — Converts all incoming channel payloads into a unified KairosEvent.

Flow:
  channel (voice/telegram/webui/cron) → normalize_*() → KairosEvent → orchestrator

Design rule:
  All upstream inputs MUST pass through normalize_*() before entering the system.
  No component downstream of this layer should ever construct a KairosEvent directly.

What this file does NOT do:
  - No business logic
  - No routing or decision-making  
  - No external API calls
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal


LOCAL_USER_ID = "local"


# ─── canonical event ──────────────────────────────────────────────────────────

@dataclass
class KairosEvent:
    channel:    Literal["voice", "telegram", "webui", "cron"]       # "voice" | "telegram" | "webui" | "cron"
    text:       str                                                 # transcribed or raw text
    user_id:    str                                                 # always the single user — you
    session_id: str                                                     # uuid, persists within a conversation
    timestamp:  float                                               # unix time
    modality:   Literal["voice", "text"]                            # "voice" | "text"
    raw:        dict = field(default_factory=dict)                  # original payload, for debugging


# ─── internal helpers ─────────────────────────────────────────────────────────

def _now() -> float:
    """Single time source. Swap to datetime.now(UTC) here if needed later."""
    return time.time()

def _make_event(
    channel:    Literal["voice", "telegram", "webui", "cron"],
    text:       str,
    session_id: str,
    modality:   Literal["voice", "text"],
    user_id:    str = LOCAL_USER_ID,
    timestamp:  float | None = None,
    raw:        dict | None = None,
) -> KairosEvent:
    """
    Central factory. All normalize_* functions go through here.
    Enforces validation in one place — not scattered across four functions.
    """
    if not text or not text.strip():
        raise ValueError(f"[normalizer] Empty text received on channel={channel!r}")

    return KairosEvent(
        channel=channel,
        text=text.strip(),
        user_id=user_id,
        session_id=session_id,
        timestamp=timestamp if timestamp is not None else _now(),
        modality=modality,
        raw=raw or {},
    )




# ─── public normalizers (thin adapters, not constructors) ─────────────────────
def normalize_telegram(update) -> KairosEvent:
    """
    Convert a python-telegram-bot Update into a KairosEvent.

    Handles text + captions (photos with text). Voice notes, stickers, and
    other media types are not supported yet — extend by checking
    update.message.voice, update.message.document, etc.

    Refs:
      https://python-telegram-bot.readthedocs.io/en/stable/telegram.message.html
      https://core.telegram.org/constructor/message
    """
    text = update.message.text or update.message.caption or ""

    return _make_event(
        channel="telegram",
        text=text,
        session_id=str(update.message.chat_id),   # chat_id = natural session boundary
        modality="text",
        user_id=str(update.message.from_user.id), # overrides LOCAL_USER_ID — real Telegram ID
        timestamp=update.message.date.timestamp(), # server time, not local clock
        raw={"update_id": update.update_id, "chat_id": update.message.chat_id},
    )

def normalize_voice(text: str, session_id: str) -> KairosEvent:
    """
    Convert a Pipecat/Deepgram transcript into a KairosEvent.
    `text` is the final transcript — not a streaming partial.
    """
    return _make_event(
        channel="voice",
        text=text,
        session_id=session_id,
        modality="voice",
        raw={"source": "deepgram"},
    )

def normalize_webui(text: str, session_id: str) -> KairosEvent:
    """Convert a WebSocket message from the browser UI into a KairosEvent."""
    return _make_event(
        channel="webui",
        text=text,
        session_id=session_id,
        modality="text",
        raw={"source": "websocket"},
    )

def normalize_cron(text: str, job_id: str | None = None) -> KairosEvent:
    """
    Wrap a scheduled/proactive trigger as a KairosEvent.

    Each cron trigger gets its own session — cron jobs don't share
    conversation history with each other or with the user's live sessions.

    `job_id` is the APScheduler job ID if available, useful for tracing
    which scheduled job fired in logs.
    """
    return _make_event(
        channel="cron",
        text=text,
        session_id=f"cron-{uuid.uuid4()}",
        modality="text",
        raw={"trigger": "scheduled", "job_id": job_id or "unknown"},
    )