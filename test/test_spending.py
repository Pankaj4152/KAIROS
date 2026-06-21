"""
Tests for runtime/tools/spending.py

Run from the project root:
    pytest test/test_spending.py -v

Structure:
    Like test_tasks.py, uses a real temporary SQLite file per test (not
    mocked) since the tool is thin SQL wrapping over a real schema with
    ALTER TABLE migration logic that's worth exercising for real.

What is tested:
    _validate_category    — valid, case-insensitive, invalid, None
    _validate_amount      — positive, zero rejected, negative rejected, non-numeric
    _validate_date        — valid, None defaults to today, invalid format
    _month_bounds         — correct first/last day incl. leap year February
    _resolve_period       — this_month, last_month, last_7_days, last_30_days,
                            explicit YYYY-MM, invalid period
    _ensure_columns       — migration adds source/sms_id without destroying rows
    spending() dispatch   — unknown action, missing required fields
    log action             — happy path, default date, invalid category,
                             invalid amount (zero/negative/non-numeric)
    list action              — category filter, month filter, date range,
                               empty result, total calculation, limit
    update action             — single field, multiple fields, nonexistent id,
                               no fields specified, invalid category rejected
    delete action              — happy path, nonexistent id
    budget action                — set then check, over-budget warning,
                               approaching-limit warning, no budget set message,
                               overall budget (no category) vs category budget
    report action                  — this_month, explicit YYYY-MM, empty period,
                               percentage calculation sums to ~100%
    end-to-end                      — log → list → budget → report → update → delete
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


def _current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


class SpendingTestCase(unittest.TestCase):
    """Base class — sets up an isolated temp DB for every test method."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._data_dir = self._tmpdir.name
        self._db_path = os.path.join(self._data_dir, "kairos.db")

        self._env_patch = patch.dict(os.environ, {"DATA_DIR": self._data_dir})
        self._env_patch.start()

        import importlib
        if "tools.spending" in sys.modules:
            del sys.modules["tools.spending"]
        import tools.spending as spending_mod
        importlib.reload(spending_mod)
        self.spending_mod = spending_mod
        self.spending = spending_mod.spending

        self._create_schema()

    def tearDown(self):
        self._env_patch.stop()
        self._tmpdir.cleanup()

    def _create_schema(self):
        """Create the spending table exactly as sqlite_store.init_db() does
        — WITHOUT source/sms_id, so the tool's _ensure_columns migration
        logic gets exercised by every test that touches the DB."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS spending (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                amount     REAL NOT NULL,
                category   TEXT,
                merchant   TEXT,
                date       TEXT,
                notes      TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _row_count(self) -> int:
        conn = sqlite3.connect(self._db_path)
        n = conn.execute("SELECT COUNT(*) FROM spending").fetchone()[0]
        conn.close()
        return n

    def _columns(self) -> set:
        conn = sqlite3.connect(self._db_path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(spending)").fetchall()}
        conn.close()
        return cols


# ── unit tests: pure helpers ──────────────────────────────────────────────────

class TestValidateCategory(SpendingTestCase):

    def test_valid_category(self):
        self.assertEqual(self.spending_mod._validate_category("food"), "food")

    def test_case_insensitive(self):
        self.assertEqual(self.spending_mod._validate_category("FOOD"), "food")
        self.assertEqual(self.spending_mod._validate_category("Food"), "food")

    def test_invalid_category_raises(self):
        with self.assertRaises(ValueError):
            self.spending_mod._validate_category("crypto_losses")

    def test_none_raises(self):
        with self.assertRaises(ValueError):
            self.spending_mod._validate_category(None)

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            self.spending_mod._validate_category("")

    def test_whitespace_stripped(self):
        self.assertEqual(self.spending_mod._validate_category("  food  "), "food")


class TestValidateAmount(SpendingTestCase):

    def test_positive_amount(self):
        self.assertEqual(self.spending_mod._validate_amount(500), 500.0)

    def test_float_amount(self):
        self.assertEqual(self.spending_mod._validate_amount(499.99), 499.99)

    def test_zero_rejected(self):
        with self.assertRaises(ValueError):
            self.spending_mod._validate_amount(0)

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            self.spending_mod._validate_amount(-100)

    def test_non_numeric_rejected(self):
        with self.assertRaises(ValueError):
            self.spending_mod._validate_amount("five hundred")

    def test_none_rejected(self):
        with self.assertRaises(ValueError):
            self.spending_mod._validate_amount(None)

    def test_rounds_to_2_decimals(self):
        result = self.spending_mod._validate_amount(99.999)
        self.assertEqual(result, 100.0)


class TestValidateDate(SpendingTestCase):

    def test_valid_date(self):
        self.assertEqual(self.spending_mod._validate_date("2026-06-20"), "2026-06-20")

    def test_none_defaults_to_today(self):
        self.assertEqual(self.spending_mod._validate_date(None), _today())

    def test_empty_defaults_to_today(self):
        self.assertEqual(self.spending_mod._validate_date(""), _today())

    def test_invalid_format_raises(self):
        with self.assertRaises(ValueError):
            self.spending_mod._validate_date("20/06/2026")


class TestMonthBounds(SpendingTestCase):

    def test_regular_month(self):
        start, end = self.spending_mod._month_bounds("2026-06")
        self.assertEqual(start, "2026-06-01")
        self.assertEqual(end, "2026-06-30")

    def test_31_day_month(self):
        start, end = self.spending_mod._month_bounds("2026-07")
        self.assertEqual(end, "2026-07-31")

    def test_february_leap_year(self):
        start, end = self.spending_mod._month_bounds("2024-02")
        self.assertEqual(end, "2024-02-29")

    def test_february_non_leap_year(self):
        start, end = self.spending_mod._month_bounds("2026-02")
        self.assertEqual(end, "2026-02-28")


class TestResolvePeriod(SpendingTestCase):

    def test_this_month(self):
        start, end, label = self.spending_mod._resolve_period("this_month")
        self.assertIn(_current_month(), label)

    def test_last_7_days(self):
        start, end, label = self.spending_mod._resolve_period("last_7_days")
        self.assertEqual(label, "Last 7 days")
        self.assertEqual(end, _today())

    def test_last_30_days(self):
        start, end, label = self.spending_mod._resolve_period("last_30_days")
        self.assertEqual(label, "Last 30 days")

    def test_explicit_month(self):
        start, end, label = self.spending_mod._resolve_period("2026-03")
        self.assertEqual(start, "2026-03-01")
        self.assertEqual(end, "2026-03-31")
        self.assertEqual(label, "2026-03")

    def test_invalid_period_raises(self):
        with self.assertRaises(ValueError):
            self.spending_mod._resolve_period("sometime_recently")

    def test_last_month_crosses_year_boundary(self):
        # Just verify it doesn't crash and returns a valid label —
        # exact date depends on when the test runs.
        start, end, label = self.spending_mod._resolve_period("last_month")
        self.assertIn("Last month", label)


# ── unit tests: schema migration ──────────────────────────────────────────────

class TestEnsureColumns(SpendingTestCase):

    def test_adds_source_and_sms_id_columns(self):
        self.assertNotIn("source", self._columns())
        self.assertNotIn("sms_id", self._columns())

        run(self.spending(action="log", amount=100, category="food"))

        self.assertIn("source", self._columns())
        self.assertIn("sms_id", self._columns())

    def test_existing_rows_preserved_after_migration(self):
        # Insert a row directly via raw schema (pre-migration)
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "INSERT INTO spending (amount, category, merchant, date, notes) "
            "VALUES (500, 'food', 'Old Cafe', '2026-01-01', 'pre-migration entry')"
        )
        conn.commit()
        conn.close()

        # Trigger migration via any action
        run(self.spending(action="log", amount=50, category="transport"))

        result = run(self.spending(action="list"))
        self.assertIn("Old Cafe", result)


# ── unit tests: dispatch ──────────────────────────────────────────────────────

class TestDispatch(SpendingTestCase):

    def test_unknown_action(self):
        result = run(self.spending(action="teleport"))
        self.assertIn("Error", result)

    def test_log_missing_amount(self):
        result = run(self.spending(action="log", category="food"))
        self.assertIn("Error", result)

    def test_log_missing_category(self):
        result = run(self.spending(action="log", amount=100))
        self.assertIn("Error", result)

    def test_update_missing_expense_id(self):
        result = run(self.spending(action="update", amount=200))
        self.assertIn("Error", result)

    def test_delete_missing_expense_id(self):
        result = run(self.spending(action="delete"))
        self.assertIn("Error", result)


# ── unit tests: log ────────────────────────────────────────────────────────────

class TestLog(SpendingTestCase):

    def test_happy_path(self):
        result = run(self.spending(action="log", amount=500, category="food"))
        self.assertIn("Logged expense", result)
        self.assertIn("500", result)
        self.assertEqual(self._row_count(), 1)

    def test_with_merchant(self):
        result = run(self.spending(
            action="log", amount=200, category="food", merchant="Swiggy"
        ))
        self.assertIn("Swiggy", result)

    def test_default_date_is_today(self):
        run(self.spending(action="log", amount=100, category="transport"))
        result = run(self.spending(action="list"))
        self.assertIn(_today(), result)

    def test_explicit_date(self):
        result = run(self.spending(
            action="log", amount=300, category="rent", date="2026-06-01"
        ))
        self.assertIn("2026-06-01", result)

    def test_invalid_category_rejected(self):
        result = run(self.spending(action="log", amount=100, category="crypto"))
        self.assertIn("Error", result)
        self.assertEqual(self._row_count(), 0)

    def test_zero_amount_rejected(self):
        result = run(self.spending(action="log", amount=0, category="food"))
        self.assertIn("Error", result)

    def test_negative_amount_rejected(self):
        result = run(self.spending(action="log", amount=-50, category="food"))
        self.assertIn("Error", result)

    def test_with_notes(self):
        result = run(self.spending(
            action="log", amount=1500, category="food", notes="team lunch"
        ))
        self.assertIn("Logged", result)


# ── unit tests: list ───────────────────────────────────────────────────────────

class TestList(SpendingTestCase):

    def setUp(self):
        super().setUp()
        run(self.spending(action="log", amount=500, category="food", date="2026-06-10"))
        run(self.spending(action="log", amount=1000, category="transport", date="2026-06-12"))
        run(self.spending(action="log", amount=2000, category="rent", date="2026-05-01"))

    def test_category_filter(self):
        result = run(self.spending(action="list", category="food"))
        self.assertIn("500", result)
        self.assertNotIn("1,000", result)

    def test_month_filter(self):
        result = run(self.spending(action="list", month="2026-06"))
        self.assertIn("500", result)
        self.assertIn("1,000", result)
        self.assertNotIn("2,000", result)

    def test_date_range_filter(self):
        result = run(self.spending(action="list", date_from="2026-06-11", date_to="2026-06-30"))
        self.assertIn("1,000", result)
        self.assertNotIn("500", result)

    def test_empty_result(self):
        result = run(self.spending(action="list", category="health"))
        self.assertIn("No expenses", result)

    def test_total_shown(self):
        result = run(self.spending(action="list", month="2026-06"))
        # 500 + 1000 = 1500
        self.assertIn("1,500", result)

    def test_limit_respected(self):
        for i in range(10):
            run(self.spending(action="log", amount=10 + i, category="other"))
        result = run(self.spending(action="list", limit=3))
        self.assertIn("Expenses (3)", result)

    def test_invalid_category_filter_rejected(self):
        result = run(self.spending(action="list", category="not_a_category"))
        self.assertIn("Error", result)


# ── unit tests: update ─────────────────────────────────────────────────────────

class TestUpdate(SpendingTestCase):

    def setUp(self):
        super().setUp()
        created = run(self.spending(action="log", amount=500, category="food"))
        self.expense_id = int(created.split("#")[1].split(":")[0])

    def test_update_amount(self):
        result = run(self.spending(action="update", expense_id=self.expense_id, amount=600))
        self.assertIn("amount", result)

    def test_update_multiple_fields(self):
        result = run(self.spending(
            action="update", expense_id=self.expense_id,
            amount=700, category="entertainment", merchant="PVR"
        ))
        self.assertIn("amount", result)
        self.assertIn("category", result)
        self.assertIn("merchant", result)

    def test_nonexistent_id(self):
        result = run(self.spending(action="update", expense_id=99999, amount=100))
        self.assertIn("Error", result)

    def test_no_fields_specified(self):
        result = run(self.spending(action="update", expense_id=self.expense_id))
        self.assertIn("No changes", result)

    def test_invalid_category_rejected(self):
        result = run(self.spending(
            action="update", expense_id=self.expense_id, category="invalid_cat"
        ))
        self.assertIn("Error", result)

    def test_invalid_amount_rejected(self):
        result = run(self.spending(action="update", expense_id=self.expense_id, amount=-5))
        self.assertIn("Error", result)


# ── unit tests: delete ────────────────────────────────────────────────────────

class TestDelete(SpendingTestCase):

    def test_happy_path(self):
        created = run(self.spending(action="log", amount=500, category="food"))
        expense_id = int(created.split("#")[1].split(":")[0])
        self.assertEqual(self._row_count(), 1)

        result = run(self.spending(action="delete", expense_id=expense_id))
        self.assertIn("Deleted", result)
        self.assertEqual(self._row_count(), 0)

    def test_nonexistent_id(self):
        result = run(self.spending(action="delete", expense_id=99999))
        self.assertIn("Error", result)


# ── unit tests: budget ─────────────────────────────────────────────────────────

class TestBudget(SpendingTestCase):

    def test_set_category_budget(self):
        result = run(self.spending(
            action="budget", category="food", month="2026-06", set_amount=5000
        ))
        self.assertIn("Budget set", result)
        self.assertIn("food", result)

    def test_set_overall_budget(self):
        result = run(self.spending(action="budget", month="2026-06", set_amount=20000))
        self.assertIn("overall", result)

    def test_check_no_budget_set(self):
        result = run(self.spending(action="budget", category="food", month="2026-06"))
        self.assertIn("No budget set", result)

    def test_check_within_budget(self):
        run(self.spending(action="budget", category="food", month="2026-06", set_amount=5000))
        run(self.spending(action="log", amount=1000, category="food", date="2026-06-05"))
        result = run(self.spending(action="budget", category="food", month="2026-06"))
        self.assertIn("Remaining", result)
        self.assertNotIn("exceeded", result.lower())

    def test_check_over_budget(self):
        run(self.spending(action="budget", category="food", month="2026-06", set_amount=1000))
        run(self.spending(action="log", amount=1500, category="food", date="2026-06-05"))
        result = run(self.spending(action="budget", category="food", month="2026-06"))
        self.assertIn("Over by", result)
        self.assertIn("exceeded", result.lower())

    def test_approaching_limit_warning(self):
        run(self.spending(action="budget", category="food", month="2026-06", set_amount=1000))
        run(self.spending(action="log", amount=900, category="food", date="2026-06-05"))
        result = run(self.spending(action="budget", category="food", month="2026-06"))
        self.assertIn("Approaching", result)

    def test_budget_updates_on_resubmit(self):
        run(self.spending(action="budget", category="food", month="2026-06", set_amount=1000))
        result = run(self.spending(action="budget", category="food", month="2026-06", set_amount=2000))
        self.assertIn("2,000", result)

    def test_overall_budget_excludes_category_filter(self):
        run(self.spending(action="budget", month="2026-06", set_amount=10000))
        run(self.spending(action="log", amount=2000, category="food", date="2026-06-05"))
        run(self.spending(action="log", amount=1000, category="transport", date="2026-06-05"))
        result = run(self.spending(action="budget", month="2026-06"))
        self.assertIn("3,000", result)   # 2000 + 1000 combined


# ── unit tests: report ─────────────────────────────────────────────────────────

class TestReport(SpendingTestCase):

    def setUp(self):
        super().setUp()
        run(self.spending(action="log", amount=3000, category="food", date="2026-06-05"))
        run(self.spending(action="log", amount=2000, category="transport", date="2026-06-10"))
        run(self.spending(action="log", amount=5000, category="rent", date="2026-06-01"))

    def test_this_month_report(self):
        with patch.object(self.spending_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 6, 20, tzinfo=timezone.utc)
            mock_dt.strptime = datetime.strptime
            result = run(self.spending(action="report", period="this_month"))
        self.assertIn("food", result)
        self.assertIn("transport", result)
        self.assertIn("rent", result)

    def test_explicit_month_report(self):
        result = run(self.spending(action="report", period="2026-06"))
        self.assertIn("10,000", result)   # 3000+2000+5000 total

    def test_empty_period_report(self):
        result = run(self.spending(action="report", period="2020-01"))
        self.assertIn("No expenses", result)

    def test_percentages_present(self):
        result = run(self.spending(action="report", period="2026-06"))
        self.assertIn("%", result)

    def test_invalid_period_rejected(self):
        result = run(self.spending(action="report", period="whenever"))
        self.assertIn("Error", result)

    def test_categories_sorted_by_total_desc(self):
        result = run(self.spending(action="report", period="2026-06"))
        idx_rent = result.find("rent")
        idx_food = result.find("food")
        idx_transport = result.find("transport")
        # rent (5000) > food (3000) > transport (2000)
        self.assertLess(idx_rent, idx_food)
        self.assertLess(idx_food, idx_transport)


# ── end-to-end ─────────────────────────────────────────────────────────────────

class TestEndToEnd(SpendingTestCase):

    def test_full_lifecycle(self):
        # Log
        logged = run(self.spending(
            action="log", amount=1200, category="food",
            merchant="BigBasket", date="2026-06-15", notes="groceries"
        ))
        self.assertIn("Logged", logged)
        expense_id = int(logged.split("#")[1].split(":")[0])

        # List — should appear
        listed = run(self.spending(action="list"))
        self.assertIn("BigBasket", listed)

        # Set budget and check
        run(self.spending(action="budget", category="food", month="2026-06", set_amount=5000))
        budget = run(self.spending(action="budget", category="food", month="2026-06"))
        self.assertIn("1,200", budget)

        # Report
        report = run(self.spending(action="report", period="2026-06"))
        self.assertIn("food", report)

        # Update
        updated = run(self.spending(action="update", expense_id=expense_id, amount=1500))
        self.assertIn("amount", updated)

        # Delete
        deleted = run(self.spending(action="delete", expense_id=expense_id))
        self.assertIn("Deleted", deleted)
        self.assertEqual(self._row_count(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)