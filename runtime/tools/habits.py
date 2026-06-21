"""
Habits tool — create habits, check them in, view streaks and consistency.

Reuses the SAME `habits` table already created by memory/sqlite_store.init_db():
    id, name, last_done, streak, target_frequency
This tool ADDS one companion table, `habit_logs`, via guarded CREATE TABLE
IF NOT EXISTS (never destroys existing rows in `habits`):
    id, habit_id, date

Why a separate log table instead of trusting the `streak` counter alone:
    A bare running counter on `habits.streak` can desync from reality if a
    checkin is backfilled for a past date, undone, or if the app crashes
    mid-update. `habit_logs` is the source of truth — one row per
    (habit_id, date) checkin. Streaks, consistency %, and the heatmap are
    all computed from this log, then `habits.streak` / `habits.last_done`
    are kept as a denormalized cache so orchestrator.py's existing
    `_fetch_habits_block()` (which reads `habits` directly) still works
    without any changes.

Frequency model (fixed enum, not a free string):
    daily      — expected every day
    weekdays   — expected Monday–Friday only
    3x_week    — expected 3 times per week, any days
    5x_week    — expected 5 times per week, any days
    weekly     — expected once per week

    This keeps streak/consistency math well-defined. The original spec's
    free-text target_frequency ("3x/week") is normalized to these values.

Actions:
    create        — add a new habit
    list          — list all habits with current streak + last checkin
    checkin       — log a completion for today (or a specific past date)
    undo_checkin  — remove a checkin (correct a mistake)
    stats         — streak, consistency %, last 30-day heatmap for one habit
    delete        — permanently remove a habit and its log history

Env vars required: none.
"""

import asyncio
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

DATA_DIR = os.getenv("DATA_DIR", "./data")
DB_PATH  = os.path.join(DATA_DIR, "kairos.db")

_VALID_FREQUENCIES = ("daily", "weekdays", "3x_week", "5x_week", "weekly")

_TARGETS_PER_WEEK = {
    "daily":    7,
    "weekdays": 5,
    "3x_week":  3,
    "5x_week":  5,
    "weekly":   1,
}


# ── connection ─────────────────────────────────────────────────────────────────

@contextmanager
def _get_conn():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        conn.close()


def _ensure_log_table(conn: sqlite3.Connection) -> None:
    """Additive companion table — never touches existing `habits` rows."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS habit_logs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            habit_id  INTEGER NOT NULL,
            date      TEXT NOT NULL,
            UNIQUE(habit_id, date)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_habit_logs_habit_date
        ON habit_logs(habit_id, date)
    """)
    conn.commit()


@contextmanager
def _conn_ready():
    with _get_conn() as conn:
        _ensure_log_table(conn)
        yield conn


# ── validation ─────────────────────────────────────────────────────────────────

def _validate_frequency(value: str | None) -> str:
    value = (value or "daily").strip().lower()
    if value not in _VALID_FREQUENCIES:
        raise ValueError(
            f"target_frequency must be one of {_VALID_FREQUENCIES} — got {value!r}"
        )
    return value


def _validate_date(value: str | None) -> str:
    if not value:
        return _today()
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"date must be YYYY-MM-DD, got {value!r}")
    return value


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _date_obj(date_str: str):
    return datetime.strptime(date_str, "%Y-%m-%d").date()


# ── streak / consistency computation ──────────────────────────────────────────

def _compute_daily_streak(log_dates: set[str], weekdays_only: bool) -> int:
    """
    Count consecutive days (or weekdays) with a checkin, walking backward
    from today (or yesterday, if today has no checkin yet — still "on streak"
    until the day actually lapses).
    """
    today = datetime.now(timezone.utc).date()
    cursor = today

    # If today has no checkin yet, start counting from yesterday so the
    # streak isn't considered broken until the day is actually over.
    if cursor.isoformat() not in log_dates:
        cursor = cursor - timedelta(days=1)

    streak = 0
    while True:
        if weekdays_only and cursor.weekday() >= 5:   # Sat=5, Sun=6 — skip
            cursor -= timedelta(days=1)
            continue
        if cursor.isoformat() in log_dates:
            streak += 1
            cursor -= timedelta(days=1)
        else:
            break
    return streak


def _compute_weekly_streak(log_dates: set[str], target_per_week: int) -> int:
    """Count consecutive ISO weeks (Mon-Sun) meeting the per-week target."""
    today = datetime.now(timezone.utc).date()
    # Start of current week (Monday)
    week_start = today - timedelta(days=today.weekday())

    streak = 0
    while True:
        week_end = week_start + timedelta(days=6)
        count = sum(
            1 for d in log_dates
            if week_start.isoformat() <= d <= week_end.isoformat()
        )
        # Current (incomplete) week doesn't break the streak even if short so far
        if week_start == today - timedelta(days=today.weekday()) and count < target_per_week:
            week_start -= timedelta(days=7)
            continue
        if count >= target_per_week:
            streak += 1
            week_start -= timedelta(days=7)
        else:
            break
    return streak


def _compute_consistency(log_dates: set[str], frequency: str, window_days: int = 30) -> float:
    """Percentage of expected occurrences actually completed in the last N days."""
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=window_days - 1)

    if frequency == "weekdays":
        expected = sum(
            1 for i in range(window_days)
            if (start + timedelta(days=i)).weekday() < 5
        )
    elif frequency == "daily":
        expected = window_days
    else:
        per_week = _TARGETS_PER_WEEK[frequency]
        expected = round(window_days / 7 * per_week)

    if expected <= 0:
        return 0.0

    actual = sum(
        1 for d in log_dates
        if start.isoformat() <= d <= today.isoformat()
    )
    return min(100.0, (actual / expected) * 100)


def _heatmap(log_dates: set[str], days: int = 30) -> str:
    """ASCII heatmap of the last N days: ■ = done, · = not done."""
    today = datetime.now(timezone.utc).date()
    cells = []
    for i in range(days - 1, -1, -1):
        d = today - timedelta(days=i)
        cells.append("■" if d.isoformat() in log_dates else "·")
    return "".join(cells)


# ── action handlers ───────────────────────────────────────────────────────────

def _do_create(name, target_frequency) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("name must not be empty")
    freq = _validate_frequency(target_frequency)

    with _conn_ready() as conn:
        existing = conn.execute(
            "SELECT id FROM habits WHERE name = ?", (name,)
        ).fetchone()
        if existing:
            return f"Error: a habit named '{name}' already exists (#{existing['id']})."

        cur = conn.execute(
            "INSERT INTO habits (name, last_done, streak, target_frequency) "
            "VALUES (?, NULL, 0, ?)",
            (name, freq),
        )
        conn.commit()
        habit_id = cur.lastrowid

    return f"Created habit #{habit_id}: \"{name}\" (target: {freq})"


def _do_list() -> str:
    with _conn_ready() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM habits ORDER BY name ASC").fetchall()]

    if not rows:
        return "No habits tracked yet."

    today = _today()
    lines = [f"Habits ({len(rows)}):"]
    for r in rows:
        last = r["last_done"] or "never"
        flag = " ⚠ not done today" if last != today else " ✓ done today"
        lines.append(
            f"  #{r['id']:<4} {r['name']:<24} streak: {r['streak']:>3}  "
            f"target: {r['target_frequency']:<9} last: {last}{flag}"
        )
    return "\n".join(lines)


def _do_checkin(habit_id, date) -> str:
    if habit_id is None:
        raise ValueError("habit_id is required")
    date = _validate_date(date)

    with _conn_ready() as conn:
        habit = conn.execute("SELECT * FROM habits WHERE id = ?", (habit_id,)).fetchone()
        if habit is None:
            return f"Error: no habit found with id {habit_id}."

        existing_log = conn.execute(
            "SELECT 1 FROM habit_logs WHERE habit_id = ? AND date = ?",
            (habit_id, date),
        ).fetchone()
        if existing_log:
            return f"Habit '{habit['name']}' already checked in for {date}."

        conn.execute(
            "INSERT INTO habit_logs (habit_id, date) VALUES (?, ?)",
            (habit_id, date),
        )

        # Recompute streak + last_done cache from the full log
        log_dates = {
            row["date"] for row in conn.execute(
                "SELECT date FROM habit_logs WHERE habit_id = ?", (habit_id,)
            ).fetchall()
        }
        freq = habit["target_frequency"] or "daily"
        if freq in ("daily", "weekdays"):
            new_streak = _compute_daily_streak(log_dates, weekdays_only=(freq == "weekdays"))
        else:
            new_streak = _compute_weekly_streak(log_dates, _TARGETS_PER_WEEK[freq])

        latest_date = max(log_dates) if log_dates else None
        conn.execute(
            "UPDATE habits SET streak = ?, last_done = ? WHERE id = ?",
            (new_streak, latest_date, habit_id),
        )
        conn.commit()

    return f"Checked in '{habit['name']}' for {date}. Current streak: {new_streak}."


def _do_undo_checkin(habit_id, date) -> str:
    if habit_id is None:
        raise ValueError("habit_id is required")
    date = _validate_date(date)

    with _conn_ready() as conn:
        habit = conn.execute("SELECT * FROM habits WHERE id = ?", (habit_id,)).fetchone()
        if habit is None:
            return f"Error: no habit found with id {habit_id}."

        deleted = conn.execute(
            "DELETE FROM habit_logs WHERE habit_id = ? AND date = ?",
            (habit_id, date),
        )
        if deleted.rowcount == 0:
            return f"No checkin found for '{habit['name']}' on {date}."

        log_dates = {
            row["date"] for row in conn.execute(
                "SELECT date FROM habit_logs WHERE habit_id = ?", (habit_id,)
            ).fetchall()
        }
        freq = habit["target_frequency"] or "daily"
        if freq in ("daily", "weekdays"):
            new_streak = _compute_daily_streak(log_dates, weekdays_only=(freq == "weekdays"))
        else:
            new_streak = _compute_weekly_streak(log_dates, _TARGETS_PER_WEEK[freq])

        latest_date = max(log_dates) if log_dates else None
        conn.execute(
            "UPDATE habits SET streak = ?, last_done = ? WHERE id = ?",
            (new_streak, latest_date, habit_id),
        )
        conn.commit()

    return f"Removed checkin for '{habit['name']}' on {date}. Current streak: {new_streak}."


def _do_stats(habit_id) -> str:
    if habit_id is None:
        raise ValueError("habit_id is required")

    with _conn_ready() as conn:
        habit = conn.execute("SELECT * FROM habits WHERE id = ?", (habit_id,)).fetchone()
        if habit is None:
            return f"Error: no habit found with id {habit_id}."

        log_dates = {
            row["date"] for row in conn.execute(
                "SELECT date FROM habit_logs WHERE habit_id = ?", (habit_id,)
            ).fetchall()
        }

    freq = habit["target_frequency"] or "daily"
    consistency = _compute_consistency(log_dates, freq, window_days=30)
    heatmap = _heatmap(log_dates, days=30)
    total_checkins = len(log_dates)

    lines = [
        f"Stats for '{habit['name']}' (#{habit_id}):",
        f"  Target:           {freq}",
        f"  Current streak:   {habit['streak']}",
        f"  Last checkin:     {habit['last_done'] or 'never'}",
        f"  Total checkins:   {total_checkins}",
        f"  30-day consistency: {consistency:.0f}%",
        "",
        f"  Last 30 days: {heatmap}",
        "  (oldest → newest, ■ = done, · = missed)",
    ]
    return "\n".join(lines)


def _do_delete(habit_id) -> str:
    if habit_id is None:
        raise ValueError("habit_id is required")

    with _conn_ready() as conn:
        habit = conn.execute("SELECT * FROM habits WHERE id = ?", (habit_id,)).fetchone()
        if habit is None:
            return f"Error: no habit found with id {habit_id}."
        conn.execute("DELETE FROM habit_logs WHERE habit_id = ?", (habit_id,))
        conn.execute("DELETE FROM habits WHERE id = ?", (habit_id,))
        conn.commit()

    return f"Deleted habit #{habit_id}: \"{habit['name']}\" (and its checkin history)."


# ── public async entrypoint ───────────────────────────────────────────────────

async def habits(
    action: str,
    habit_id: int | None = None,
    name: str | None = None,
    target_frequency: str | None = None,
    date: str | None = None,
) -> str:
    """
    Track habits: create, check in, view streaks/consistency, and delete.

    Args:
        action:           which operation to perform (see below). Required.
        habit_id:         target habit ID. Required for checkin/undo_checkin/stats/delete.
        name:             habit name. Required for create, e.g. "meditation", "gym".
        target_frequency: one of 'daily', 'weekdays', '3x_week', '5x_week', 'weekly'.
                          Defaults to 'daily' for create.
        date:             ISO date 'YYYY-MM-DD' for checkin/undo_checkin.
                          Defaults to today.

    Actions:
        create        — add a new habit. Requires: name. Optional: target_frequency.
        list          — list all habits with current streak and last checkin date.
        checkin       — log a completion. Requires: habit_id. Defaults to today;
                        pass date to backfill a past day.
        undo_checkin  — remove a checkin (fix a mistake). Requires: habit_id.
                        Defaults to today.
        stats         — streak, 30-day consistency %, and ASCII heatmap for one
                        habit. Requires: habit_id.
        delete        — permanently remove a habit and all its checkin history.
                        Requires: habit_id.

    Returns a plain string in all cases. Never raises to the caller.
    """
    action = (action or "").strip().lower()
    valid_actions = {"create", "list", "checkin", "undo_checkin", "stats", "delete"}
    if action not in valid_actions:
        return f"Error: Unknown action '{action}'. Valid: {', '.join(sorted(valid_actions))}."

    try:
        if action == "create":
            return await asyncio.to_thread(_do_create, name, target_frequency)

        if action == "list":
            return await asyncio.to_thread(_do_list)

        if action == "checkin":
            return await asyncio.to_thread(_do_checkin, habit_id, date)

        if action == "undo_checkin":
            return await asyncio.to_thread(_do_undo_checkin, habit_id, date)

        if action == "stats":
            return await asyncio.to_thread(_do_stats, habit_id)

        if action == "delete":
            return await asyncio.to_thread(_do_delete, habit_id)

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.exception("habits() failed for action=%r", action)
        return f"Error: Habits tool failed unexpectedly — {type(e).__name__}: {e}"

    return "Error: Unhandled action path."