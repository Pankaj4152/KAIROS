"""
Tests for runtime/memory/sqlite_store.py

Covers:
  - Schema init (tables + indexes created)
  - Tasks: add, fetch open, complete, validation
  - Events: add, fetch upcoming, fetch by date, validation
  - Habits: fetch, mark done (streak increment)
  - Spending: add, summary aggregation, validation
"""

import logging
import os
import pytest

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _patch_db(tmp_data_dir, monkeypatch):
    """Point sqlite_store at a temp directory for every test."""
    db_path = os.path.join(tmp_data_dir, "kairos.db")
    monkeypatch.setattr("memory.sqlite_store.DATA_DIR", tmp_data_dir)
    monkeypatch.setattr("memory.sqlite_store.DB_PATH", db_path)
    logger.info("DB path set to %s", db_path)

    from memory.sqlite_store import init_db
    init_db()
    logger.info("Database initialized")


# ── schema ────────────────────────────────────────────────────────────────────

class TestSchemaInit:
    def test_tables_created(self):
        from memory.sqlite_store import get_conn
        with get_conn() as conn:
            tables = [
                r[0] for r in
                conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            ]
        logger.info("Tables found: %s", tables)
        for expected in ("tasks", "events", "habits", "spending", "memory_meta", "schema_version"):
            assert expected in tables, f"Missing table: {expected}"

    def test_schema_version_recorded(self):
        from memory.sqlite_store import get_conn, SCHEMA_VERSION
        with get_conn() as conn:
            version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        logger.info("Schema version: %d (expected %d)", version, SCHEMA_VERSION)
        assert version == SCHEMA_VERSION


# ── tasks ─────────────────────────────────────────────────────────────────────

class TestTasks:
    def test_add_and_fetch(self):
        from memory.sqlite_store import add_task, fetch_open_tasks
        row_id = add_task("Buy groceries", due_date="2026-04-01", priority=3)
        logger.info("Created task id=%d", row_id)
        assert row_id > 0

        tasks = fetch_open_tasks()
        logger.info("Open tasks: %s", tasks)
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Buy groceries"
        assert tasks[0]["priority"] == 3

    def test_complete_task(self):
        from memory.sqlite_store import add_task, fetch_open_tasks, complete_task
        tid = add_task("Finish report")
        complete_task(tid)
        tasks = fetch_open_tasks()
        logger.info("Open tasks after completing id=%d: %d remaining", tid, len(tasks))
        assert len(tasks) == 0

    def test_priority_sorting(self):
        from memory.sqlite_store import add_task, fetch_open_tasks
        add_task("Low prio", priority=1)
        add_task("High prio", priority=3)
        add_task("Normal prio", priority=2)

        tasks = fetch_open_tasks()
        priorities = [t["priority"] for t in tasks]
        logger.info("Task priorities (should be descending): %s", priorities)
        assert priorities == sorted(priorities, reverse=True)

    def test_empty_title_rejected(self):
        from memory.sqlite_store import add_task
        with pytest.raises(ValueError, match="title must not be empty"):
            add_task("")
        logger.info("Empty title correctly rejected")

    def test_bad_date_format_rejected(self):
        from memory.sqlite_store import add_task
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            add_task("Task", due_date="April 1")
        logger.info("Bad date format correctly rejected")

    def test_invalid_priority_rejected(self):
        from memory.sqlite_store import add_task
        with pytest.raises(ValueError, match="priority must be 1, 2, or 3"):
            add_task("Task", priority=5)
        logger.info("Invalid priority correctly rejected")


# ── events ────────────────────────────────────────────────────────────────────

class TestEvents:
    def test_add_and_fetch_upcoming(self):
        from memory.sqlite_store import add_event, fetch_upcoming_events
        eid = add_event("Team standup", start_time="2099-01-01T09:00:00")
        logger.info("Created event id=%d", eid)
        assert eid > 0

        events = fetch_upcoming_events()
        logger.info("Upcoming events: %d found", len(events))
        assert any(e["title"] == "Team standup" for e in events)

    def test_fetch_events_for_date(self):
        from memory.sqlite_store import add_event, fetch_events_for_date
        add_event("Morning jog", start_time="2026-05-15T07:00:00")
        add_event("Dinner", start_time="2026-05-15T19:00:00")
        add_event("Next day", start_time="2026-05-16T10:00:00")

        events = fetch_events_for_date("2026-05-15")
        titles = [e["title"] for e in events]
        logger.info("Events on 2026-05-15: %s", titles)
        assert len(events) == 2
        assert "Next day" not in titles

    def test_empty_title_rejected(self):
        from memory.sqlite_store import add_event
        with pytest.raises(ValueError, match="title must not be empty"):
            add_event("", start_time="2026-01-01T09:00:00")
        logger.info("Empty event title correctly rejected")

    def test_bad_datetime_rejected(self):
        from memory.sqlite_store import add_event
        with pytest.raises(ValueError, match="ISO 8601"):
            add_event("Meeting", start_time="tomorrow 9am")
        logger.info("Bad datetime correctly rejected")


# ── habits ────────────────────────────────────────────────────────────────────

class TestHabits:
    def _insert_habit(self, name="Meditate", streak=0):
        from memory.sqlite_store import get_conn
        with get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO habits (name, streak) VALUES (?, ?)", (name, streak)
            )
            conn.commit()
            return cur.lastrowid

    def test_fetch_habits(self):
        from memory.sqlite_store import fetch_habits
        self._insert_habit("Meditate", streak=5)
        self._insert_habit("Exercise", streak=12)

        habits = fetch_habits()
        names = [h["name"] for h in habits]
        logger.info("Habits: %s", names)
        assert len(habits) == 2
        assert "Meditate" in names

    def test_mark_habit_done_increments_streak(self):
        from memory.sqlite_store import fetch_habits, mark_habit_done
        hid = self._insert_habit("Read", streak=3)

        mark_habit_done(hid)
        habits = fetch_habits()
        habit = next(h for h in habits if h["id"] == hid)
        logger.info("Streak after mark_done: %d (expected 4)", habit["streak"])
        assert habit["streak"] == 4
        assert habit["last_done"] is not None


# ── spending ──────────────────────────────────────────────────────────────────

class TestSpending:
    def test_add_and_summary(self):
        from memory.sqlite_store import add_spending, fetch_spending_summary
        add_spending(15.50, "food", merchant="Cafe")
        add_spending(42.00, "food", merchant="Restaurant")
        add_spending(9.99, "transport")

        summary = fetch_spending_summary()
        logger.info("Spending summary: %s", summary)
        categories = {s["category"]: s["total"] for s in summary}
        assert categories["food"] == pytest.approx(57.50)
        assert categories["transport"] == pytest.approx(9.99)

    def test_negative_amount_rejected(self):
        from memory.sqlite_store import add_spending
        with pytest.raises(ValueError, match="must be positive"):
            add_spending(-10, "food")
        logger.info("Negative amount correctly rejected")

    def test_zero_amount_rejected(self):
        from memory.sqlite_store import add_spending
        with pytest.raises(ValueError, match="must be positive"):
            add_spending(0, "food")
        logger.info("Zero amount correctly rejected")

    def test_empty_category_rejected(self):
        from memory.sqlite_store import add_spending
        with pytest.raises(ValueError, match="category must not be empty"):
            add_spending(10, "")
        logger.info("Empty category correctly rejected")
