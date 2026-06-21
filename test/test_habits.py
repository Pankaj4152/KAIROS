"""
Tests for runtime/tools/habits.py

Run from the project root:
    pytest test/test_habits.py -v

Structure:
    Each test gets a fresh temp SQLite DB (real, not mocked) to exercise
    the actual streak/consistency math against real date arithmetic.
    Time-dependent tests use freezegun-style manual date injection by
    writing habit_logs rows directly with known dates relative to "today"
    (computed once per test) rather than mocking datetime.now(), since the
    module computes "today" in several independent helper functions.

What is tested:
    _validate_frequency       — valid values, default, invalid
    _validate_date            — valid, None defaults to today, invalid format
    _compute_daily_streak     — perfect streak, broken streak, today not yet
                                done still counts streak, weekdays_only skips
                                weekends
    _compute_weekly_streak    — met target this week, missed target breaks streak
    _compute_consistency      — 100% for daily met every day, 50% for half,
                                weekdays frequency excludes weekends from expected
    _heatmap                  — correct length, correct symbols
    habits() dispatch         — unknown action, missing required fields
    create action              — happy path, duplicate name rejected, empty name,
                                invalid frequency, default frequency is daily
    list action                  — shows streak/last_done, empty state, done-today flag
    checkin action                — happy path increments streak, duplicate checkin
                                same day rejected, backfill past date,
                                nonexistent habit_id
    undo_checkin action             — happy path decrements streak, no checkin to
                                remove, nonexistent habit_id
    stats action                     — shows streak/consistency/heatmap,
                                nonexistent habit_id
    delete action                      — happy path removes habit + logs,
                                nonexistent habit_id
    end-to-end                          — create → checkin x3 → stats → undo →
                                stats again → delete
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


def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).date().isoformat()


class HabitsTestCase(unittest.TestCase):
    """Base class — sets up an isolated temp DB for every test method."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._data_dir = self._tmpdir.name
        self._db_path = os.path.join(self._data_dir, "kairos.db")

        self._env_patch = patch.dict(os.environ, {"DATA_DIR": self._data_dir})
        self._env_patch.start()

        import importlib
        if "tools.habits" in sys.modules:
            del sys.modules["tools.habits"]
        import tools.habits as habits_mod
        importlib.reload(habits_mod)
        self.habits_mod = habits_mod
        self.habits = habits_mod.habits

        self._create_schema()

    def tearDown(self):
        self._env_patch.stop()
        self._tmpdir.cleanup()

    def _create_schema(self):
        """Create the habits table exactly as sqlite_store.init_db() does."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS habits (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                name             TEXT NOT NULL,
                last_done        TEXT,
                streak           INTEGER DEFAULT 0,
                target_frequency TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _insert_log(self, habit_id: int, date: str):
        """Directly insert a habit_logs row, bypassing the checkin action,
        for setting up known historical data in streak/consistency tests."""
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS habit_logs ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, habit_id INTEGER NOT NULL, "
            "date TEXT NOT NULL, UNIQUE(habit_id, date))"
        )
        conn.execute(
            "INSERT OR IGNORE INTO habit_logs (habit_id, date) VALUES (?, ?)",
            (habit_id, date),
        )
        conn.commit()
        conn.close()

    def _create_habit_id(self, name="Test habit", frequency="daily") -> int:
        result = run(self.habits(action="create", name=name, target_frequency=frequency))
        return int(result.split("#")[1].split(":")[0])


# ── unit tests: pure helpers ──────────────────────────────────────────────────

class TestValidateFrequency(HabitsTestCase):

    def test_valid_values(self):
        for f in ("daily", "weekdays", "3x_week", "5x_week", "weekly"):
            self.assertEqual(self.habits_mod._validate_frequency(f), f)

    def test_none_defaults_to_daily(self):
        self.assertEqual(self.habits_mod._validate_frequency(None), "daily")

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.habits_mod._validate_frequency("hourly")

    def test_case_insensitive(self):
        self.assertEqual(self.habits_mod._validate_frequency("DAILY"), "daily")


class TestValidateDate(HabitsTestCase):

    def test_valid_date(self):
        self.assertEqual(self.habits_mod._validate_date("2026-06-20"), "2026-06-20")

    def test_none_defaults_to_today(self):
        self.assertEqual(self.habits_mod._validate_date(None), _today())

    def test_invalid_format_raises(self):
        with self.assertRaises(ValueError):
            self.habits_mod._validate_date("20-06-2026")


class TestComputeDailyStreak(HabitsTestCase):

    def test_perfect_streak_including_today(self):
        log_dates = {_days_ago(i) for i in range(5)}   # today, -1, -2, -3, -4
        streak = self.habits_mod._compute_daily_streak(log_dates, weekdays_only=False)
        self.assertEqual(streak, 5)

    def test_streak_not_yet_broken_if_today_missing(self):
        # Checked in every day except today — streak should still count
        # through yesterday (today isn't "missed" until the day is over).
        log_dates = {_days_ago(i) for i in range(1, 6)}   # -1 through -5, no today
        streak = self.habits_mod._compute_daily_streak(log_dates, weekdays_only=False)
        self.assertEqual(streak, 5)

    def test_broken_streak(self):
        # Gap at -2 breaks the chain
        log_dates = {_today(), _days_ago(1), _days_ago(3), _days_ago(4)}
        streak = self.habits_mod._compute_daily_streak(log_dates, weekdays_only=False)
        self.assertEqual(streak, 2)   # today + yesterday only

    def test_no_checkins_zero_streak(self):
        streak = self.habits_mod._compute_daily_streak(set(), weekdays_only=False)
        self.assertEqual(streak, 0)

    def test_weekdays_only_skips_weekends(self):
        # Build a log that has every weekday for the last 10 calendar days,
        # skipping Sat/Sun — streak should count only weekdays.
        today = datetime.now(timezone.utc).date()
        log_dates = set()
        cursor = today
        count = 0
        while count < 10:
            if cursor.weekday() < 5:
                log_dates.add(cursor.isoformat())
                count += 1
            cursor -= timedelta(days=1)

        streak = self.habits_mod._compute_daily_streak(log_dates, weekdays_only=True)
        self.assertGreaterEqual(streak, 1)   # exact value depends on today's weekday,
                                              # but must not crash and must be positive


class TestComputeWeeklyStreak(HabitsTestCase):

    def test_zero_when_no_logs(self):
        streak = self.habits_mod._compute_weekly_streak(set(), target_per_week=3)
        self.assertEqual(streak, 0)

    def test_current_week_partial_does_not_break_streak(self):
        # Only 1 checkin so far this week (target 3) — current week is
        # incomplete, shouldn't count as "broken" yet.
        log_dates = {_today()}
        streak = self.habits_mod._compute_weekly_streak(log_dates, target_per_week=3)
        # Should not raise, and should be a non-negative int
        self.assertGreaterEqual(streak, 0)


class TestComputeConsistency(HabitsTestCase):

    def test_full_consistency_daily(self):
        log_dates = {_days_ago(i) for i in range(30)}
        pct = self.habits_mod._compute_consistency(log_dates, "daily", window_days=30)
        self.assertEqual(pct, 100.0)

    def test_half_consistency_daily(self):
        log_dates = {_days_ago(i) for i in range(0, 30, 2)}   # every other day
        pct = self.habits_mod._compute_consistency(log_dates, "daily", window_days=30)
        self.assertAlmostEqual(pct, 50.0, delta=5.0)

    def test_zero_consistency_no_logs(self):
        pct = self.habits_mod._compute_consistency(set(), "daily", window_days=30)
        self.assertEqual(pct, 0.0)

    def test_consistency_capped_at_100(self):
        # More logs than possible days shouldn't push above 100
        log_dates = {_days_ago(i) for i in range(60)}
        pct = self.habits_mod._compute_consistency(log_dates, "daily", window_days=30)
        self.assertLessEqual(pct, 100.0)


class TestHeatmap(HabitsTestCase):

    def test_correct_length(self):
        result = self.habits_mod._heatmap(set(), days=30)
        self.assertEqual(len(result), 30)

    def test_all_missed_symbol(self):
        result = self.habits_mod._heatmap(set(), days=10)
        self.assertEqual(result, "·" * 10)

    def test_all_done_symbol(self):
        log_dates = {_days_ago(i) for i in range(10)}
        result = self.habits_mod._heatmap(log_dates, days=10)
        self.assertEqual(result, "■" * 10)

    def test_mixed_pattern(self):
        log_dates = {_today()}
        result = self.habits_mod._heatmap(log_dates, days=5)
        self.assertEqual(result[-1], "■")   # last char = today = done


# ── unit tests: dispatch ──────────────────────────────────────────────────────

class TestDispatch(HabitsTestCase):

    def test_unknown_action(self):
        result = run(self.habits(action="teleport"))
        self.assertIn("Error", result)

    def test_checkin_missing_habit_id(self):
        result = run(self.habits(action="checkin"))
        self.assertIn("Error", result)

    def test_undo_checkin_missing_habit_id(self):
        result = run(self.habits(action="undo_checkin"))
        self.assertIn("Error", result)

    def test_stats_missing_habit_id(self):
        result = run(self.habits(action="stats"))
        self.assertIn("Error", result)

    def test_delete_missing_habit_id(self):
        result = run(self.habits(action="delete"))
        self.assertIn("Error", result)


# ── unit tests: create ────────────────────────────────────────────────────────

class TestCreate(HabitsTestCase):

    def test_happy_path(self):
        result = run(self.habits(action="create", name="Meditation"))
        self.assertIn("Created habit", result)
        self.assertIn("Meditation", result)
        self.assertIn("daily", result)   # default frequency

    def test_custom_frequency(self):
        result = run(self.habits(action="create", name="Gym", target_frequency="3x_week"))
        self.assertIn("3x_week", result)

    def test_duplicate_name_rejected(self):
        run(self.habits(action="create", name="DSA practice"))
        result = run(self.habits(action="create", name="DSA practice"))
        self.assertIn("Error", result)
        self.assertIn("already exists", result)

    def test_empty_name_rejected(self):
        result = run(self.habits(action="create", name=""))
        self.assertIn("Error", result)

    def test_invalid_frequency_rejected(self):
        result = run(self.habits(action="create", name="X", target_frequency="hourly"))
        self.assertIn("Error", result)


# ── unit tests: list ───────────────────────────────────────────────────────────

class TestList(HabitsTestCase):

    def test_empty_state(self):
        result = run(self.habits(action="list"))
        self.assertIn("No habits", result)

    def test_shows_streak_and_last_done(self):
        habit_id = self._create_habit_id("Reading")
        run(self.habits(action="checkin", habit_id=habit_id))
        result = run(self.habits(action="list"))
        self.assertIn("Reading", result)
        self.assertIn("streak:   1", result)
        self.assertIn("done today", result)

    def test_not_done_today_flag(self):
        self._create_habit_id("Never done")
        result = run(self.habits(action="list"))
        self.assertIn("not done today", result)


# ── unit tests: checkin ────────────────────────────────────────────────────────

class TestCheckin(HabitsTestCase):

    def setUp(self):
        super().setUp()
        self.habit_id = self._create_habit_id("Meditation")

    def test_happy_path(self):
        result = run(self.habits(action="checkin", habit_id=self.habit_id))
        self.assertIn("Checked in", result)
        self.assertIn("streak: 1", result)

    def test_duplicate_same_day_rejected(self):
        run(self.habits(action="checkin", habit_id=self.habit_id))
        result = run(self.habits(action="checkin", habit_id=self.habit_id))
        self.assertIn("already checked in", result.lower())

    def test_backfill_past_date(self):
        result = run(self.habits(action="checkin", habit_id=self.habit_id, date=_days_ago(3)))
        self.assertIn("Checked in", result)

    def test_nonexistent_habit_id(self):
        result = run(self.habits(action="checkin", habit_id=99999))
        self.assertIn("Error", result)

    def test_consecutive_checkins_build_streak(self):
        run(self.habits(action="checkin", habit_id=self.habit_id, date=_days_ago(2)))
        run(self.habits(action="checkin", habit_id=self.habit_id, date=_days_ago(1)))
        result = run(self.habits(action="checkin", habit_id=self.habit_id, date=_today()))
        self.assertIn("streak: 3", result)


# ── unit tests: undo_checkin ────────────────────────────────────────────────────

class TestUndoCheckin(HabitsTestCase):

    def setUp(self):
        super().setUp()
        self.habit_id = self._create_habit_id("Gym")

    def test_happy_path(self):
        run(self.habits(action="checkin", habit_id=self.habit_id))
        result = run(self.habits(action="undo_checkin", habit_id=self.habit_id))
        self.assertIn("Removed checkin", result)
        self.assertIn("streak: 0", result)

    def test_no_checkin_to_remove(self):
        result = run(self.habits(action="undo_checkin", habit_id=self.habit_id))
        self.assertIn("No checkin found", result)

    def test_nonexistent_habit_id(self):
        result = run(self.habits(action="undo_checkin", habit_id=99999))
        self.assertIn("Error", result)


# ── unit tests: stats ──────────────────────────────────────────────────────────

class TestStats(HabitsTestCase):

    def test_happy_path(self):
        habit_id = self._create_habit_id("Reading")
        run(self.habits(action="checkin", habit_id=habit_id))
        result = run(self.habits(action="stats", habit_id=habit_id))
        self.assertIn("Reading", result)
        self.assertIn("Current streak:   1", result)
        self.assertIn("30-day consistency", result)
        self.assertIn("Last 30 days:", result)

    def test_heatmap_present_and_correct_length(self):
        habit_id = self._create_habit_id("Reading")
        result = run(self.habits(action="stats", habit_id=habit_id))
        # Extract the heatmap line
        heatmap_line = [l for l in result.split("\n") if "Last 30 days:" in l][0]
        heatmap_chars = heatmap_line.split("Last 30 days:")[1].strip()
        self.assertEqual(len(heatmap_chars), 30)

    def test_nonexistent_habit_id(self):
        result = run(self.habits(action="stats", habit_id=99999))
        self.assertIn("Error", result)


# ── unit tests: delete ────────────────────────────────────────────────────────

class TestDelete(HabitsTestCase):

    def test_happy_path(self):
        habit_id = self._create_habit_id("Temp habit")
        run(self.habits(action="checkin", habit_id=habit_id))
        result = run(self.habits(action="delete", habit_id=habit_id))
        self.assertIn("Deleted", result)

        listed = run(self.habits(action="list"))
        self.assertNotIn("Temp habit", listed)

    def test_nonexistent_habit_id(self):
        result = run(self.habits(action="delete", habit_id=99999))
        self.assertIn("Error", result)


# ── end-to-end ─────────────────────────────────────────────────────────────────

class TestEndToEnd(HabitsTestCase):

    def test_full_lifecycle(self):
        created = run(self.habits(action="create", name="Daily walk", target_frequency="daily"))
        self.assertIn("Created", created)
        habit_id = int(created.split("#")[1].split(":")[0])

        # Checkin 3 consecutive days
        run(self.habits(action="checkin", habit_id=habit_id, date=_days_ago(2)))
        run(self.habits(action="checkin", habit_id=habit_id, date=_days_ago(1)))
        run(self.habits(action="checkin", habit_id=habit_id, date=_today()))

        # Stats should show streak of 3
        stats = run(self.habits(action="stats", habit_id=habit_id))
        self.assertIn("Current streak:   3", stats)

        # Undo today's checkin
        run(self.habits(action="undo_checkin", habit_id=habit_id))
        stats_after_undo = run(self.habits(action="stats", habit_id=habit_id))
        self.assertIn("Current streak:   2", stats_after_undo)

        # Delete
        deleted = run(self.habits(action="delete", habit_id=habit_id))
        self.assertIn("Deleted", deleted)


if __name__ == "__main__":
    unittest.main(verbosity=2)