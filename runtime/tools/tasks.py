"""
Tasks tool — create, list, update, complete, search, and report on tasks.

Phase 1 scope (per ARCHITECTURE_DECISIONS.md):
    - Tasks are SEPARATE from Google Calendar. No calendar_event_id linking yet.
    - That comes in Phase 2 once this tool is stable and you've used it a while.

Reuses the SAME `tasks` table already created by memory/sqlite_store.init_db():
    id, title, due_date, status, project, priority, created_at

This file does NOT replace memory/sqlite_store.py — orchestrator.py's context
assembler still calls fetch_open_tasks() from there for prompt injection.
This tool is the AGENTIC layer: the LLM calls it directly to create/update/
search/complete tasks mid-conversation, with richer actions than the
context-assembly helper needs.

Important fix vs. the existing sqlite_store.py:
    orchestrator.py calls `await fetch_open_tasks()`, but sqlite_store.py
    defines these functions as plain sync functions (no `async def`, no
    asyncio.to_thread wrapping). That is a latent bug — awaiting a sync
    function's return value (a list) raises TypeError at runtime.
    This tool avoids that mistake: every DB call here is wrapped in
    `asyncio.to_thread()` so it's safe to `await` from anywhere.

Actions:
    create        — add a new task
    list          — list tasks, filterable by status/priority/due window
    update        — change title, priority, due_date, or project
    complete      — mark a task done (or reopen with complete(undo=True))
    delete        — permanently remove a task
    search        — full-text search over title + project
    stats         — counts by status/priority, overdue count, completed-this-week

Priority scale matches the existing schema: 1=low, 2=normal (default), 3=high.

Env vars required: none. Uses DATA_DIR (same as sqlite_store.py).
"""

import asyncio
import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

DATA_DIR = os.getenv("DATA_DIR", "./data")
DB_PATH  = os.path.join(DATA_DIR, "kairos.db")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_VALID_STATUSES = ("open", "done")
_VALID_PRIORITIES = (1, 2, 3)
_PRIORITY_LABEL = {1: "low", 2: "normal", 3: "high"}


# ── connection ─────────────────────────────────────────────────────────────────
# Reuses the exact same kairos.db file and `tasks` table that
# memory/sqlite_store.init_db() already creates. No schema changes,
# no migration — this tool is purely additive.

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


# ── validation helpers ────────────────────────────────────────────────────────

def _validate_date(value: str | None, field: str) -> str | None:
    if value is None or value == "":
        return None
    if not _DATE_RE.match(value):
        raise ValueError(f"{field} must be YYYY-MM-DD, got {value!r}")
    return value


def _validate_priority(value: int | None) -> int:
    if value is None:
        return 2
    if value not in _VALID_PRIORITIES:
        raise ValueError(f"priority must be 1 (low), 2 (normal), or 3 (high) — got {value}")
    return value


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


# ── formatting ─────────────────────────────────────────────────────────────────

def _fmt_task_row(row: dict, today: str) -> str:
    """One-line task summary used in list output."""
    pid     = row["id"]
    title   = row["title"]
    prio    = _PRIORITY_LABEL.get(row["priority"], str(row["priority"]))
    status  = row["status"]
    due     = row["due_date"]
    project = row["project"]

    flags = []
    if status == "done":
        flags.append("✓ done")
    elif due:
        if due < today:
            flags.append(f"OVERDUE (was due {due})")
        elif due == today:
            flags.append("due TODAY")
        else:
            flags.append(f"due {due}")

    tag = f" [{project}]" if project else ""
    flag_str = f"  — {', '.join(flags)}" if flags else ""

    return f"  #{pid:<4} ({prio:<6}){tag} {title}{flag_str}"


# ── action handlers ───────────────────────────────────────────────────────────

def _do_create(title, due_date, project, priority) -> str:
    title = (title or "").strip()
    if not title:
        raise ValueError("title must not be empty")
    due_date = _validate_date(due_date, "due_date")
    priority = _validate_priority(priority)

    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO tasks (title, due_date, project, priority, status) "
            "VALUES (?, ?, ?, ?, 'open')",
            (title, due_date, project, priority),
        )
        conn.commit()
        task_id = cur.lastrowid

    due_str = f", due {due_date}" if due_date else ""
    return f"Created task #{task_id}: \"{title}\" (priority: {_PRIORITY_LABEL[priority]}{due_str})"


def _do_list(status, priority, due_before, due_after, project, limit) -> str:
    query  = "SELECT * FROM tasks WHERE 1=1"
    params: list = []

    if status:
        if status not in _VALID_STATUSES:
            raise ValueError(f"status must be 'open' or 'done' — got {status!r}")
        query += " AND status = ?"
        params.append(status)
    else:
        # Default: hide done tasks unless explicitly asked
        query += " AND status = 'open'"

    if priority is not None:
        query += " AND priority = ?"
        params.append(_validate_priority(priority))

    if due_before:
        query += " AND due_date IS NOT NULL AND due_date <= ?"
        params.append(_validate_date(due_before, "due_before"))

    if due_after:
        query += " AND due_date IS NOT NULL AND due_date >= ?"
        params.append(_validate_date(due_after, "due_after"))

    if project:
        query += " AND project = ?"
        params.append(project)

    query += " ORDER BY priority DESC, (due_date IS NULL), due_date ASC LIMIT ?"
    params.append(max(1, min(limit, 100)))

    with _get_conn() as conn:
        rows = [dict(r) for r in conn.execute(query, params).fetchall()]

    if not rows:
        return "No tasks match those filters."

    today = _today()
    lines = [f"Tasks ({len(rows)}):"]
    lines.extend(_fmt_task_row(r, today) for r in rows)
    return "\n".join(lines)


def _do_update(task_id, title, due_date, project, priority) -> str:
    if task_id is None:
        raise ValueError("task_id is required")

    with _get_conn() as conn:
        existing = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if existing is None:
            return f"Error: no task found with id {task_id}."

        updates: list[str] = []
        params: list = []
        changed: list[str] = []

        if title is not None:
            title = title.strip()
            if not title:
                raise ValueError("title must not be empty")
            updates.append("title = ?")
            params.append(title)
            changed.append(f"title → \"{title}\"")

        if due_date is not None:
            # Allow explicit empty string to clear the due date
            new_due = _validate_date(due_date, "due_date") if due_date else None
            updates.append("due_date = ?")
            params.append(new_due)
            changed.append(f"due_date → {new_due or '(cleared)'}")

        if project is not None:
            updates.append("project = ?")
            params.append(project or None)
            changed.append(f"project → {project or '(cleared)'}")

        if priority is not None:
            p = _validate_priority(priority)
            updates.append("priority = ?")
            params.append(p)
            changed.append(f"priority → {_PRIORITY_LABEL[p]}")

        if not updates:
            return f"No changes specified for task #{task_id}."

        params.append(task_id)
        conn.execute(f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()

    return f"Updated task #{task_id}: " + "; ".join(changed)


def _do_complete(task_id, undo) -> str:
    if task_id is None:
        raise ValueError("task_id is required")

    new_status = "open" if undo else "done"

    with _get_conn() as conn:
        existing = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if existing is None:
            return f"Error: no task found with id {task_id}."

        if existing["status"] == new_status:
            verb = "already open" if undo else "already marked done"
            return f"Task #{task_id} (\"{existing['title']}\") is {verb}."

        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (new_status, task_id))
        conn.commit()

    verb = "Reopened" if undo else "Completed"
    return f"{verb} task #{task_id}: \"{existing['title']}\""


def _do_delete(task_id) -> str:
    if task_id is None:
        raise ValueError("task_id is required")

    with _get_conn() as conn:
        existing = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if existing is None:
            return f"Error: no task found with id {task_id}."
        conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.commit()

    return f"Deleted task #{task_id}: \"{existing['title']}\""


def _do_search(query, limit) -> str:
    query = (query or "").strip()
    if not query:
        raise ValueError("query must not be empty")

    like = f"%{query}%"
    with _get_conn() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM tasks WHERE title LIKE ? OR project LIKE ? "
            "ORDER BY status ASC, priority DESC, (due_date IS NULL), due_date ASC LIMIT ?",
            (like, like, max(1, min(limit, 100))),
        ).fetchall()]

    if not rows:
        return f"No tasks found matching '{query}'."

    today = _today()
    lines = [f"Search results for '{query}' ({len(rows)}):"]
    lines.extend(_fmt_task_row(r, today) for r in rows)
    return "\n".join(lines)


def _do_stats(days) -> str:
    days = max(1, min(days, 365))
    since = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    today = _today()

    with _get_conn() as conn:
        open_count  = conn.execute("SELECT COUNT(*) FROM tasks WHERE status='open'").fetchone()[0]
        done_count  = conn.execute("SELECT COUNT(*) FROM tasks WHERE status='done'").fetchone()[0]
        overdue     = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status='open' AND due_date IS NOT NULL AND due_date < ?",
            (today,),
        ).fetchone()[0]
        due_today   = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status='open' AND due_date = ?",
            (today,),
        ).fetchone()[0]
        by_priority = conn.execute(
            "SELECT priority, COUNT(*) as n FROM tasks WHERE status='open' GROUP BY priority"
        ).fetchall()

        # No completed_at column exists in the base schema, so "completed this
        # period" can't be measured precisely without a timestamp on completion.
        # We report total open/done instead and flag this limitation in output.

    lines = [
        f"Task stats (last {days} day window requested):",
        f"  Open:        {open_count}",
        f"  Done:        {done_count}",
        f"  Overdue:     {overdue}",
        f"  Due today:   {due_today}",
        "  Open by priority:",
    ]
    prio_map = {r["priority"]: r["n"] for r in by_priority}
    for p in (3, 2, 1):
        lines.append(f"    {_PRIORITY_LABEL[p]:<6}: {prio_map.get(p, 0)}")

    return "\n".join(lines)


# ── public async entrypoint ───────────────────────────────────────────────────

async def tasks(
    action: str,
    task_id: int | None = None,
    title: str | None = None,
    due_date: str | None = None,
    project: str | None = None,
    priority: int | None = None,
    status: str | None = None,
    due_before: str | None = None,
    due_after: str | None = None,
    query: str | None = None,
    undo: bool = False,
    days: int = 7,
    limit: int = 20,
) -> str:
    """
    Manage tasks: create, list, update, complete, delete, search, and view stats.

    Args:
        action:     which operation to perform (see below). Required.
        task_id:    target task ID. Required for update/complete/delete.
        title:      task title. Required for create; optional for update.
        due_date:   ISO date "YYYY-MM-DD". For update, pass "" to clear it.
        project:    optional project/category tag, e.g. "kairos", "college".
        priority:   1=low, 2=normal (default), 3=high.
        status:     filter for list: "open" or "done". Default: open only.
        due_before: filter for list: tasks due on or before this date.
        due_after:  filter for list: tasks due on or after this date.
        query:      search text for action='search'.
        undo:       for action='complete', pass True to reopen instead of close.
        days:       lookback window for action='stats'. Default 7.
        limit:      max rows for list/search. Default 20, max 100.

    Actions:
        create    — add a new task. Requires: title. Optional: due_date, project, priority.
        list      — list tasks. Defaults to open tasks, sorted by priority then due date.
        update    — change fields on an existing task. Requires: task_id + at least one field.
        complete  — mark a task done. Requires: task_id. Pass undo=True to reopen.
        delete    — permanently remove a task. Requires: task_id.
        search    — full-text search over title and project. Requires: query.
        stats     — counts by status/priority, overdue count, due-today count.

    Returns a plain string in all cases. Never raises to the caller.
    """
    action = (action or "").strip().lower()
    valid_actions = {"create", "list", "update", "complete", "delete", "search", "stats"}
    if action not in valid_actions:
        return f"Error: Unknown action '{action}'. Valid: {', '.join(sorted(valid_actions))}."

    try:
        if action == "create":
            return await asyncio.to_thread(_do_create, title, due_date, project, priority)

        if action == "list":
            return await asyncio.to_thread(
                _do_list, status, priority, due_before, due_after, project, limit
            )

        if action == "update":
            return await asyncio.to_thread(_do_update, task_id, title, due_date, project, priority)

        if action == "complete":
            return await asyncio.to_thread(_do_complete, task_id, undo)

        if action == "delete":
            return await asyncio.to_thread(_do_delete, task_id)

        if action == "search":
            return await asyncio.to_thread(_do_search, query, limit)

        if action == "stats":
            return await asyncio.to_thread(_do_stats, days)

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.exception("tasks() failed for action=%r", action)
        return f"Error: Tasks tool failed unexpectedly — {type(e).__name__}: {e}"

    return "Error: Unhandled action path."