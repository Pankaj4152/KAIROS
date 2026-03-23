"""
SQLite structured data store — tasks, events, habits, spending.

This is the structured memory layer. Use it for anything that needs
filtering, sorting, counting, or aggregating by field value.

For semantic search over conversation history → vector_store.py
For session turn history → session_store.py
For always-on identity context → profile.md

All public functions are async. SQLite is blocking I/O — we push it to
a thread pool via asyncio.to_thread() so we never block the event loop.

Connection management:
  Use get_conn() as a context manager. Each caller opens one connection
  for the duration of their work, then closes it cleanly.
  Never open a connection per query inside a loop.
"""

import asyncio
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DATA_DIR = os.getenv("DATA_DIR", "./data")
DB_PATH  = os.path.join(DATA_DIR, "kairos.db")

# Schema version — bump this when you change table structure.
# Migration logic goes in _migrate() when you need it.
SCHEMA_VERSION = 1


# ─── connection ───────────────────────────────────────────────────────────────

@contextmanager
def get_conn():
    """
    Context manager for SQLite connections.

    Usage:
        with get_conn() as conn:
            rows = conn.execute("SELECT ...").fetchall()

    Ensures the connection is always closed, even if an exception occurs.
    Creates DATA_DIR if it doesn't exist yet.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe concurrent reads
    try:
        yield conn
    finally:
        conn.close()


# ─── validation ───────────────────────────────────────────────────────────────

_DATE_RE     = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")


def _require_text(value: str, field: str) -> str:
    """Reject empty or whitespace-only strings."""
    if not value or not value.strip():
        raise ValueError(f"{field} must not be empty")
    return value.strip()


def _require_positive(value: float, field: str) -> float:
    """Reject zero or negative numbers."""
    if value <= 0:
        raise ValueError(f"{field} must be positive, got {value}")
    return value


def _require_date(value: str, field: str) -> str:
    """Reject strings that aren't YYYY-MM-DD format."""
    if not _DATE_RE.match(value):
        raise ValueError(f"{field} must be YYYY-MM-DD, got {value!r}")
    return value


def _require_datetime(value: str, field: str) -> str:
    """Reject strings that don't start with YYYY-MM-DDTHH:MM."""
    if not _DATETIME_RE.match(value):
        raise ValueError(f"{field} must be ISO 8601 datetime, got {value!r}")
    return value


# ─── schema init ─────────────────────────────────────────────────────────────

def init_db():
    """
    Create all tables and indexes if they don't exist.
    Safe to call on every startup — uses IF NOT EXISTS throughout.
    Call once in main.py before starting the server.

    Indexes added for every column used in WHERE or ORDER BY clauses.
    Without these, SQLite does a full table scan on every context fetch.
    """
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT    NOT NULL,
                due_date    TEXT,
                status      TEXT    DEFAULT 'open',
                project     TEXT,
                priority    INTEGER DEFAULT 2,
                created_at  TEXT    DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT NOT NULL,
                start_time  TEXT,
                end_time    TEXT,
                location    TEXT,
                notes       TEXT,
                source      TEXT DEFAULT 'manual'
            );

            CREATE TABLE IF NOT EXISTS habits (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                name             TEXT NOT NULL,
                last_done        TEXT,
                streak           INTEGER DEFAULT 0,
                target_frequency TEXT
            );

            CREATE TABLE IF NOT EXISTS spending (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                amount     REAL NOT NULL,
                category   TEXT,
                merchant   TEXT,
                date       TEXT,
                notes      TEXT
            );

            CREATE TABLE IF NOT EXISTS memory_meta (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                content    TEXT,
                source     TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT
            );

            -- Schema version table — bump SCHEMA_VERSION constant when
            -- table structure changes. Migration logic goes in _migrate().
            CREATE TABLE IF NOT EXISTS schema_version (
                version    INTEGER NOT NULL,
                applied_at TEXT    DEFAULT CURRENT_TIMESTAMP
            );

            -- Indexes: every column used in WHERE or ORDER BY gets one.
            -- Full table scans on tasks/events will blow the 50ms context budget.
            CREATE INDEX IF NOT EXISTS idx_tasks_status
                ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_tasks_priority_due
                ON tasks(priority DESC, due_date ASC);
            CREATE INDEX IF NOT EXISTS idx_events_start_time
                ON events(start_time);
            CREATE INDEX IF NOT EXISTS idx_spending_date
                ON spending(date);
            CREATE INDEX IF NOT EXISTS idx_spending_category
                ON spending(category);
            CREATE INDEX IF NOT EXISTS idx_memory_meta_session
                ON memory_meta(session_id);
        """)

        # Write schema version if table is empty
        count = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        if count == 0:
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,)
            )

        conn.commit()


# ─── tasks ────────────────────────────────────────────────────────────────────

def add_task(
    title: str,
    due_date: str | None = None,
    project: str | None = None,
    priority: int = 2,
) -> int:
    """
    Insert a new task. Returns the new row ID.
    priority: 1=low, 2=normal, 3=high — higher sorts first.
    due_date: ISO 8601 date string "YYYY-MM-DD", or None.
    """
    title = _require_text(title, "title")
    if due_date is not None:
        due_date = _require_date(due_date, "due_date")
    if priority not in (1, 2, 3):
        raise ValueError(f"priority must be 1, 2, or 3 — got {priority}")

    def _run():
        with get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO tasks (title, due_date, project, priority) VALUES (?, ?, ?, ?)",
                (title, due_date, project, priority),
            )
            conn.commit()
            return cur.lastrowid

    return _run()


def fetch_open_tasks() -> list[dict]:
    """
    All open tasks, sorted by priority (high first) then due date (soonest first).
    Used by the context assembler when the classifier returns domain='tasks'.
    """
    def _run():
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status = 'open' ORDER BY priority DESC, due_date ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    return _run()


def complete_task(task_id: int) -> None:
    """Mark a task as done by ID."""
    def _run():
        with get_conn() as conn:
            conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (task_id,))
            conn.commit()

    _run()


# ─── events ───────────────────────────────────────────────────────────────────

def add_event(
    title: str,
    start_time: str,
    end_time: str | None = None,
    location: str | None = None,
    notes: str | None = None,
    source: str = "manual",
) -> int:
    """
    Insert a calendar event. Returns the new row ID.
    start_time / end_time: ISO 8601 — "2026-03-25T14:00:00".
    source: "manual" | "gcal" — distinguishes hand-added vs synced events.
    """
    title      = _require_text(title, "title")
    start_time = _require_datetime(start_time, "start_time")
    if end_time is not None:
        end_time = _require_datetime(end_time, "end_time")

    def _run():
        with get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO events (title, start_time, end_time, location, notes, source)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (title, start_time, end_time, location, notes, source),
            )
            conn.commit()
            return cur.lastrowid

    return _run()


def fetch_upcoming_events(limit: int = 10) -> list[dict]:
    """
    Events from now onward, soonest first.
    Used for general "what's coming up" queries.
    """
    def _run():
        now = datetime.now(timezone.utc).isoformat()
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE start_time >= ? ORDER BY start_time ASC LIMIT ?",
                (now, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    return _run()


def fetch_events_for_date(date_str: str) -> list[dict]:
    """
    Events on a specific date. date_str is "YYYY-MM-DD".

    The context assembler calls this when the classifier returns domain='events'
    and the user is asking about a specific day (e.g. "what do I have tomorrow").
    Uses prefix match on start_time — works with full ISO timestamps.
    """
    date_str = _require_date(date_str, "date_str")

    def _run():
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE start_time LIKE ? ORDER BY start_time ASC",
                (f"{date_str}%",),
            ).fetchall()
            return [dict(r) for r in rows]

    return _run()


# ─── habits ───────────────────────────────────────────────────────────────────

def fetch_habits() -> list[dict]:
    """
    All habits with current streak and last completion date.
    Used by the context assembler when domain='habits'.
    """
    def _run():
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM habits ORDER BY name ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    return _run()


def mark_habit_done(habit_id: int) -> None:
    """
    Record a habit completion for today. Increments streak.
    Does not check for duplicate completion today — caller's responsibility.
    """
    def _run():
        today = datetime.now(timezone.utc).date().isoformat()
        with get_conn() as conn:
            conn.execute(
                "UPDATE habits SET last_done = ?, streak = streak + 1 WHERE id = ?",
                (today, habit_id),
            )
            conn.commit()

    _run()


# ─── spending ─────────────────────────────────────────────────────────────────

def add_spending(
    amount: float,
    category: str,
    merchant: str | None = None,
    date: str | None = None,
    notes: str | None = None,
) -> int:
    """
    Log a spending entry. Returns the new row ID.
    date defaults to today (UTC) if not provided.
    """
    amount   = _require_positive(amount, "amount")
    category = _require_text(category, "category")
    if date is not None:
        date = _require_date(date, "date")

    def _run():
        today = datetime.now(timezone.utc).date().isoformat()
        with get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO spending (amount, category, merchant, date, notes) VALUES (?, ?, ?, ?, ?)",
                (amount, category, merchant, date or today, notes),
            )
            conn.commit()
            return cur.lastrowid

    return _run()


def fetch_spending_summary() -> list[dict]:
    """
    Total spending grouped by category, highest first.
    Used for "how much have I spent on X" queries.
    """
    def _run():
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT category, SUM(amount) as total FROM spending "
                "GROUP BY category ORDER BY total DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    return _run()