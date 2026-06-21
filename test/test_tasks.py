"""
Tests for runtime/tools/tasks.py

Run from the project root:
    pytest test/test_tasks.py -v

Structure:
    All tests use a real temporary SQLite file (not mocked) since the tool
    is thin SQL wrapping — mocking sqlite3 itself would test nothing useful.
    Each test gets a fresh temp DB via setUp/tearDown so tests never
    interfere with each other or with your real data/kairos.db.

What is tested:
    _validate_date          — valid format, invalid format, None passthrough
    _validate_priority      — valid values, invalid value, None defaults to 2
    _fmt_task_row           — overdue flag, due-today flag, done flag, project tag
    tasks() dispatch        — unknown action, missing required fields
    create action           — happy path, empty title, invalid priority, invalid date
    list action              — default open-only, status filter, priority filter,
                               due_before/due_after, project filter, empty result, limit
    update action            — single field, multiple fields, clear due_date,
                               nonexistent task_id, no fields specified
    complete action           — mark done, idempotent (already done), undo/reopen,
                               nonexistent task_id
    delete action             — happy path, nonexistent task_id
    search action              — match in title, match in project, no match, empty query
    stats action                — counts open/done, overdue, due today, by priority
    end-to-end                  — create → list → update → complete → stats → delete
"""

import asyncio
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runtime"))


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _days_from_today(n: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=n)).date().isoformat()


class TasksTestCase(unittest.TestCase):
    """Base class — sets up an isolated temp DB for every test method."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._data_dir = self._tmpdir.name
        self._db_path = os.path.join(self._data_dir, "kairos.db")

        # Patch env + module-level constants before importing the module fresh
        self._env_patch = patch.dict(os.environ, {"DATA_DIR": self._data_dir})
        self._env_patch.start()

        # Import fresh each time so DATA_DIR/DB_PATH module globals pick up the patch
        import importlib
        if "tools.tasks" in sys.modules:
            del sys.modules["tools.tasks"]
        import tools.tasks as tasks_mod
        importlib.reload(tasks_mod)
        self.tasks_mod = tasks_mod
        self.tasks = tasks_mod.tasks

        self._create_schema()

    def tearDown(self):
        self._env_patch.stop()
        self._tmpdir.cleanup()

    def _create_schema(self):
        """Create the tasks table exactly as sqlite_store.init_db() does."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT    NOT NULL,
                due_date    TEXT,
                status      TEXT    DEFAULT 'open',
                project     TEXT,
                priority    INTEGER DEFAULT 2,
                created_at  TEXT    DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def _row_count(self) -> int:
        conn = sqlite3.connect(self._db_path)
        n = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        conn.close()
        return n


# ── unit tests: pure helpers ──────────────────────────────────────────────────

class TestValidateDate(TasksTestCase):

    def test_valid_date(self):
        result = self.tasks_mod._validate_date("2026-06-25", "due_date")
        self.assertEqual(result, "2026-06-25")

    def test_none_returns_none(self):
        self.assertIsNone(self.tasks_mod._validate_date(None, "due_date"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(self.tasks_mod._validate_date("", "due_date"))

    def test_invalid_format_raises(self):
        with self.assertRaises(ValueError):
            self.tasks_mod._validate_date("25-06-2026", "due_date")

    def test_invalid_format_text_raises(self):
        with self.assertRaises(ValueError):
            self.tasks_mod._validate_date("next friday", "due_date")


class TestValidatePriority(TasksTestCase):

    def test_valid_values(self):
        for p in (1, 2, 3):
            self.assertEqual(self.tasks_mod._validate_priority(p), p)

    def test_none_defaults_to_2(self):
        self.assertEqual(self.tasks_mod._validate_priority(None), 2)

    def test_invalid_value_raises(self):
        with self.assertRaises(ValueError):
            self.tasks_mod._validate_priority(5)

    def test_zero_raises(self):
        with self.assertRaises(ValueError):
            self.tasks_mod._validate_priority(0)


class TestFmtTaskRow(TasksTestCase):

    def test_overdue_flag(self):
        row = {"id": 1, "title": "Old task", "priority": 2, "status": "open",
               "due_date": "2020-01-01", "project": None}
        result = self.tasks_mod._fmt_task_row(row, _today())
        self.assertIn("OVERDUE", result)

    def test_due_today_flag(self):
        today = _today()
        row = {"id": 1, "title": "Today task", "priority": 2, "status": "open",
               "due_date": today, "project": None}
        result = self.tasks_mod._fmt_task_row(row, today)
        self.assertIn("due TODAY", result)

    def test_done_flag(self):
        row = {"id": 1, "title": "Finished", "priority": 1, "status": "done",
               "due_date": None, "project": None}
        result = self.tasks_mod._fmt_task_row(row, _today())
        self.assertIn("done", result)

    def test_project_tag_shown(self):
        row = {"id": 1, "title": "Build feature", "priority": 3, "status": "open",
               "due_date": None, "project": "kairos"}
        result = self.tasks_mod._fmt_task_row(row, _today())
        self.assertIn("[kairos]", result)

    def test_no_due_date_no_flag(self):
        row = {"id": 1, "title": "Someday task", "priority": 1, "status": "open",
               "due_date": None, "project": None}
        result = self.tasks_mod._fmt_task_row(row, _today())
        self.assertNotIn("OVERDUE", result)
        self.assertNotIn("due TODAY", result)


# ── unit tests: dispatch ──────────────────────────────────────────────────────

class TestDispatch(TasksTestCase):

    def test_unknown_action(self):
        result = run(self.tasks(action="teleport"))
        self.assertIn("Error", result)
        self.assertIn("teleport", result)

    def test_create_missing_title(self):
        result = run(self.tasks(action="create"))
        self.assertIn("Error", result)

    def test_update_missing_task_id(self):
        result = run(self.tasks(action="update", title="New title"))
        self.assertIn("Error", result)

    def test_complete_missing_task_id(self):
        result = run(self.tasks(action="complete"))
        self.assertIn("Error", result)

    def test_delete_missing_task_id(self):
        result = run(self.tasks(action="delete"))
        self.assertIn("Error", result)

    def test_search_missing_query(self):
        result = run(self.tasks(action="search"))
        self.assertIn("Error", result)


# ── unit tests: create ────────────────────────────────────────────────────────

class TestCreate(TasksTestCase):

    def test_happy_path(self):
        result = run(self.tasks(action="create", title="Finish report"))
        self.assertIn("Created task", result)
        self.assertIn("Finish report", result)
        self.assertEqual(self._row_count(), 1)

    def test_with_due_date_and_priority(self):
        due = _days_from_today(3)
        result = run(self.tasks(
            action="create", title="Submit PR", due_date=due, priority=3
        ))
        self.assertIn("high", result)
        self.assertIn(due, result)

    def test_with_project(self):
        result = run(self.tasks(action="create", title="DSA practice", project="college"))
        self.assertIn("Created task", result)

    def test_empty_title_rejected(self):
        result = run(self.tasks(action="create", title=""))
        self.assertIn("Error", result)
        self.assertEqual(self._row_count(), 0)

    def test_whitespace_title_rejected(self):
        result = run(self.tasks(action="create", title="   "))
        self.assertIn("Error", result)

    def test_invalid_priority_rejected(self):
        result = run(self.tasks(action="create", title="X", priority=9))
        self.assertIn("Error", result)
        self.assertEqual(self._row_count(), 0)

    def test_invalid_due_date_rejected(self):
        result = run(self.tasks(action="create", title="X", due_date="not-a-date"))
        self.assertIn("Error", result)

    def test_default_priority_is_normal(self):
        run(self.tasks(action="create", title="Default priority task"))
        result = run(self.tasks(action="list"))
        self.assertIn("normal", result)


# ── unit tests: list ──────────────────────────────────────────────────────────

class TestList(TasksTestCase):

    def setUp(self):
        super().setUp()
        run(self.tasks(action="create", title="Open task 1", priority=1))
        run(self.tasks(action="create", title="Open task 2", priority=3))
        task3 = run(self.tasks(action="create", title="Done task", priority=2))
        # Extract ID from "Created task #3: ..." and mark it done
        task_id = int(task3.split("#")[1].split(":")[0])
        run(self.tasks(action="complete", task_id=task_id))

    def test_default_excludes_done(self):
        result = run(self.tasks(action="list"))
        self.assertIn("Open task 1", result)
        self.assertIn("Open task 2", result)
        self.assertNotIn("Done task", result)

    def test_status_done_filter(self):
        result = run(self.tasks(action="list", status="done"))
        self.assertIn("Done task", result)
        self.assertNotIn("Open task 1", result)

    def test_priority_filter(self):
        result = run(self.tasks(action="list", priority=3))
        self.assertIn("Open task 2", result)
        self.assertNotIn("Open task 1", result)

    def test_empty_result(self):
        result = run(self.tasks(action="list", project="nonexistent_project_xyz"))
        self.assertIn("No tasks", result)

    def test_due_before_filter(self):
        run(self.tasks(action="create", title="Due soon", due_date=_days_from_today(1)))
        run(self.tasks(action="create", title="Due far", due_date=_days_from_today(30)))
        result = run(self.tasks(action="list", due_before=_days_from_today(5)))
        self.assertIn("Due soon", result)
        self.assertNotIn("Due far", result)

    def test_due_after_filter(self):
        run(self.tasks(action="create", title="Due soon2", due_date=_days_from_today(1)))
        run(self.tasks(action="create", title="Due far2", due_date=_days_from_today(30)))
        result = run(self.tasks(action="list", due_after=_days_from_today(10)))
        self.assertIn("Due far2", result)
        self.assertNotIn("Due soon2", result)

    def test_project_filter(self):
        run(self.tasks(action="create", title="Project task", project="kairos"))
        result = run(self.tasks(action="list", project="kairos"))
        self.assertIn("Project task", result)
        self.assertNotIn("Open task 1", result)

    def test_sorted_by_priority_desc(self):
        result = run(self.tasks(action="list"))
        # "Open task 2" (priority 3) should appear before "Open task 1" (priority 1)
        idx_high = result.find("Open task 2")
        idx_low  = result.find("Open task 1")
        self.assertLess(idx_high, idx_low)

    def test_limit_respected(self):
        for i in range(10):
            run(self.tasks(action="create", title=f"Bulk task {i}"))
        result = run(self.tasks(action="list", limit=3))
        # Should show "Tasks (3):" header
        self.assertIn("Tasks (3)", result)


# ── unit tests: update ────────────────────────────────────────────────────────

class TestUpdate(TasksTestCase):

    def setUp(self):
        super().setUp()
        created = run(self.tasks(action="create", title="Original title", priority=1))
        self.task_id = int(created.split("#")[1].split(":")[0])

    def test_update_title(self):
        result = run(self.tasks(action="update", task_id=self.task_id, title="New title"))
        self.assertIn("title", result)
        listed = run(self.tasks(action="list"))
        self.assertIn("New title", listed)

    def test_update_multiple_fields(self):
        result = run(self.tasks(
            action="update", task_id=self.task_id,
            title="Updated", priority=3, project="kairos"
        ))
        self.assertIn("title", result)
        self.assertIn("priority", result)
        self.assertIn("project", result)

    def test_clear_due_date(self):
        run(self.tasks(action="update", task_id=self.task_id, due_date=_days_from_today(5)))
        result = run(self.tasks(action="update", task_id=self.task_id, due_date=""))
        self.assertIn("cleared", result)

    def test_nonexistent_task_id(self):
        result = run(self.tasks(action="update", task_id=99999, title="X"))
        self.assertIn("Error", result)
        self.assertIn("no task found", result.lower())

    def test_no_fields_specified(self):
        result = run(self.tasks(action="update", task_id=self.task_id))
        self.assertIn("No changes", result)

    def test_update_empty_title_rejected(self):
        result = run(self.tasks(action="update", task_id=self.task_id, title=""))
        self.assertIn("Error", result)

    def test_update_invalid_priority_rejected(self):
        result = run(self.tasks(action="update", task_id=self.task_id, priority=99))
        self.assertIn("Error", result)


# ── unit tests: complete ──────────────────────────────────────────────────────

class TestComplete(TasksTestCase):

    def setUp(self):
        super().setUp()
        created = run(self.tasks(action="create", title="Task to complete"))
        self.task_id = int(created.split("#")[1].split(":")[0])

    def test_mark_done(self):
        result = run(self.tasks(action="complete", task_id=self.task_id))
        self.assertIn("Completed", result)
        listed = run(self.tasks(action="list", status="done"))
        self.assertIn("Task to complete", listed)

    def test_idempotent_already_done(self):
        run(self.tasks(action="complete", task_id=self.task_id))
        result = run(self.tasks(action="complete", task_id=self.task_id))
        self.assertIn("already marked done", result)

    def test_undo_reopens(self):
        run(self.tasks(action="complete", task_id=self.task_id))
        result = run(self.tasks(action="complete", task_id=self.task_id, undo=True))
        self.assertIn("Reopened", result)
        listed = run(self.tasks(action="list"))
        self.assertIn("Task to complete", listed)

    def test_undo_idempotent(self):
        result = run(self.tasks(action="complete", task_id=self.task_id, undo=True))
        self.assertIn("already open", result)

    def test_nonexistent_task_id(self):
        result = run(self.tasks(action="complete", task_id=99999))
        self.assertIn("Error", result)


# ── unit tests: delete ────────────────────────────────────────────────────────

class TestDelete(TasksTestCase):

    def test_happy_path(self):
        created = run(self.tasks(action="create", title="Delete me"))
        task_id = int(created.split("#")[1].split(":")[0])
        self.assertEqual(self._row_count(), 1)

        result = run(self.tasks(action="delete", task_id=task_id))
        self.assertIn("Deleted", result)
        self.assertEqual(self._row_count(), 0)

    def test_nonexistent_task_id(self):
        result = run(self.tasks(action="delete", task_id=99999))
        self.assertIn("Error", result)


# ── unit tests: search ─────────────────────────────────────────────────────────

class TestSearch(TasksTestCase):

    def setUp(self):
        super().setUp()
        run(self.tasks(action="create", title="Fix login bug", project="kairos"))
        run(self.tasks(action="create", title="Write documentation"))
        run(self.tasks(action="create", title="Review PR", project="kairos-backend"))

    def test_match_in_title(self):
        result = run(self.tasks(action="search", query="login"))
        self.assertIn("Fix login bug", result)
        self.assertNotIn("Write documentation", result)

    def test_match_in_project(self):
        result = run(self.tasks(action="search", query="kairos"))
        self.assertIn("Fix login bug", result)
        self.assertIn("Review PR", result)

    def test_no_match(self):
        result = run(self.tasks(action="search", query="xyznonexistent"))
        self.assertIn("No tasks found", result)

    def test_empty_query_rejected(self):
        result = run(self.tasks(action="search", query=""))
        self.assertIn("Error", result)


# ── unit tests: stats ─────────────────────────────────────────────────────────

class TestStats(TasksTestCase):

    def setUp(self):
        super().setUp()
        run(self.tasks(action="create", title="Open low", priority=1))
        run(self.tasks(action="create", title="Open high", priority=3))
        done_resp = run(self.tasks(action="create", title="Done task"))
        done_id = int(done_resp.split("#")[1].split(":")[0])
        run(self.tasks(action="complete", task_id=done_id))

    def test_open_done_counts(self):
        result = run(self.tasks(action="stats"))
        self.assertIn("Open:        2", result)
        self.assertIn("Done:        1", result)

    def test_overdue_count(self):
        run(self.tasks(action="create", title="Overdue task", due_date="2020-01-01"))
        result = run(self.tasks(action="stats"))
        self.assertIn("Overdue:     1", result)

    def test_due_today_count(self):
        run(self.tasks(action="create", title="Today task", due_date=_today()))
        result = run(self.tasks(action="stats"))
        self.assertIn("Due today:   1", result)

    def test_by_priority_breakdown(self):
        result = run(self.tasks(action="stats"))
        self.assertIn("high", result)
        self.assertIn("low", result)


# ── end-to-end ─────────────────────────────────────────────────────────────────

class TestEndToEnd(TasksTestCase):

    def test_full_lifecycle(self):
        # Create
        created = run(self.tasks(
            action="create", title="Ship feature X",
            due_date=_days_from_today(2), priority=3, project="kairos"
        ))
        self.assertIn("Created", created)
        task_id = int(created.split("#")[1].split(":")[0])

        # List — should appear
        listed = run(self.tasks(action="list"))
        self.assertIn("Ship feature X", listed)

        # Update
        updated = run(self.tasks(action="update", task_id=task_id, priority=2))
        self.assertIn("priority", updated)

        # Complete
        completed = run(self.tasks(action="complete", task_id=task_id))
        self.assertIn("Completed", completed)

        # Stats reflect completion
        stats = run(self.tasks(action="stats"))
        self.assertIn("Done:        1", stats)

        # Delete
        deleted = run(self.tasks(action="delete", task_id=task_id))
        self.assertIn("Deleted", deleted)
        self.assertEqual(self._row_count(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)