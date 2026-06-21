"""
Spending tool — log expenses, check budgets, view category/period reports.

Phase 1 scope (per ARCHITECTURE_DECISIONS.md):
    - MANUAL logging only. You tell Kairos what you spent; it logs it.
    - No UPI/bank API integration — those aren't available for retail
      access in India (PhonePe/Google Pay don't expose APIs; most banks
      don't expose retail transaction APIs either).
    - SMS-parser auto-logging is Phase 2 — NOT built here, but the schema
      below already includes a `source` column ('manual' | 'sms_parsed')
      and a `sms_id` column (nullable, unique) so Phase 2 can be added
      later purely as a new action + background job, with zero migration.

Reuses the SAME `spending` table already created by memory/sqlite_store.init_db():
    id, amount, category, merchant, date, notes
This tool ADDS two columns via guarded ALTER TABLE (safe to run repeatedly,
never destroys existing rows): `source` and `sms_id`.

Categories are a fixed, validated set (see _VALID_CATEGORIES below). The LLM
must pick one of these — if your real spending doesn't fit, "other" always
works, and notes can carry the specifics.

Actions:
    log       — record a new expense
    list      — list expenses, filterable by category/month/date range
    update    — change amount, category, merchant, date, or notes on an entry
    delete    — remove a mistakenly logged expense
    budget    — compare spend-to-date in a category/month against a budget you set
    report    — category breakdown + total for a period (this_month, last_month,
                last_7_days, last_30_days, or explicit YYYY-MM)

Env vars required: none.
"""

import asyncio
import logging
import os
import sqlite3
from calendar import monthrange
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

DATA_DIR = os.getenv("DATA_DIR", "./data")
DB_PATH  = os.path.join(DATA_DIR, "kairos.db")

# Fixed category set. "other" is always the escape hatch — exact wording
# matters since the LLM is instructed (via the registry schema enum) to
# pick from this list.
_VALID_CATEGORIES = (
    "food", "transport", "utilities", "entertainment",
    "health", "shopping", "subscriptions", "rent", "education", "other",
)

_PERIOD_PRESETS = ("this_month", "last_month", "last_7_days", "last_30_days")


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


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """
    Add Phase-2-ready columns if they don't already exist.
    Safe to call on every connection — checked against PRAGMA table_info.
    Never touches existing rows; new columns default to NULL / 'manual'.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(spending)").fetchall()}
    if "source" not in cols:
        conn.execute("ALTER TABLE spending ADD COLUMN source TEXT DEFAULT 'manual'")
    if "sms_id" not in cols:
        conn.execute("ALTER TABLE spending ADD COLUMN sms_id TEXT")
    conn.commit()


def _ensure_budgets_table(conn: sqlite3.Connection) -> None:
    """
    Small dedicated table for user-set budgets, keyed by category + month.
    month is stored as 'YYYY-MM'; NULL category = overall monthly budget.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS budgets (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            category  TEXT,
            month     TEXT NOT NULL,
            amount    REAL NOT NULL,
            UNIQUE(category, month)
        )
    """)
    conn.commit()


@contextmanager
def _conn_ready():
    """Connection with Phase-2 columns and budgets table guaranteed present."""
    with _get_conn() as conn:
        _ensure_columns(conn)
        _ensure_budgets_table(conn)
        yield conn


# ── validation ─────────────────────────────────────────────────────────────────

def _validate_category(value: str | None) -> str:
    if not value:
        raise ValueError(f"category is required. Valid: {', '.join(_VALID_CATEGORIES)}")
    value = value.strip().lower()
    if value not in _VALID_CATEGORIES:
        raise ValueError(
            f"Unknown category '{value}'. Valid: {', '.join(_VALID_CATEGORIES)}"
        )
    return value


def _validate_amount(value) -> float:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"amount must be a number, got {value!r}")
    if amount <= 0:
        raise ValueError(f"amount must be positive, got {amount}")
    return round(amount, 2)


def _validate_date(value: str | None) -> str:
    if not value:
        return datetime.now(timezone.utc).date().isoformat()
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"date must be YYYY-MM-DD, got {value!r}")
    return value


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _month_bounds(month_str: str) -> tuple[str, str]:
    """Return (first_day, last_day) ISO dates for a 'YYYY-MM' string."""
    year, mon = (int(x) for x in month_str.split("-"))
    last_day = monthrange(year, mon)[1]
    return f"{month_str}-01", f"{month_str}-{last_day:02d}"


def _resolve_period(period: str) -> tuple[str, str, str]:
    """
    Resolve a period keyword or explicit 'YYYY-MM' into (start, end, label).
    Raises ValueError for unrecognised input.
    """
    now = datetime.now(timezone.utc)

    if period == "this_month":
        month = now.strftime("%Y-%m")
        start, end = _month_bounds(month)
        return start, end, f"This month ({month})"

    if period == "last_month":
        first_of_this = now.replace(day=1)
        last_month_date = first_of_this - timedelta(days=1)
        month = last_month_date.strftime("%Y-%m")
        start, end = _month_bounds(month)
        return start, end, f"Last month ({month})"

    if period == "last_7_days":
        start = (now - timedelta(days=7)).date().isoformat()
        end   = now.date().isoformat()
        return start, end, "Last 7 days"

    if period == "last_30_days":
        start = (now - timedelta(days=30)).date().isoformat()
        end   = now.date().isoformat()
        return start, end, "Last 30 days"

    # Explicit YYYY-MM
    try:
        datetime.strptime(period + "-01", "%Y-%m-%d")
        start, end = _month_bounds(period)
        return start, end, period
    except ValueError:
        raise ValueError(
            f"Unrecognised period '{period}'. Use one of {_PERIOD_PRESETS} "
            f"or an explicit 'YYYY-MM'."
        )


# ── formatting ─────────────────────────────────────────────────────────────────

def _fmt_amount(val: float) -> str:
    return f"₹{val:,.2f}"


def _fmt_expense_row(row: dict) -> str:
    merchant = f" @ {row['merchant']}" if row["merchant"] else ""
    notes    = f"  ({row['notes']})" if row["notes"] else ""
    return f"  #{row['id']:<4} {row['date']}  {_fmt_amount(row['amount']):>12}  {row['category']:<14}{merchant}{notes}"


def _bar(pct: float, width: int = 20) -> str:
    """ASCII progress bar for budget/category visualisation."""
    pct = max(0.0, min(pct, 1.0))
    filled = round(pct * width)
    return "█" * filled + "░" * (width - filled)


# ── action handlers ───────────────────────────────────────────────────────────

def _do_log(amount, category, merchant, date, notes) -> str:
    amount   = _validate_amount(amount)
    category = _validate_category(category)
    date     = _validate_date(date)

    with _conn_ready() as conn:
        cur = conn.execute(
            "INSERT INTO spending (amount, category, merchant, date, notes, source) "
            "VALUES (?, ?, ?, ?, ?, 'manual')",
            (amount, category, merchant, date, notes),
        )
        conn.commit()
        row_id = cur.lastrowid

    merchant_str = f" at {merchant}" if merchant else ""
    return f"Logged expense #{row_id}: {_fmt_amount(amount)} ({category}){merchant_str} on {date}"


def _do_list(category, month, date_from, date_to, limit) -> str:
    query  = "SELECT * FROM spending WHERE 1=1"
    params: list = []

    if category:
        query += " AND category = ?"
        params.append(_validate_category(category))

    if month:
        start, end = _month_bounds(month)
        query += " AND date >= ? AND date <= ?"
        params.extend([start, end])
    else:
        if date_from:
            query += " AND date >= ?"
            params.append(_validate_date(date_from))
        if date_to:
            query += " AND date <= ?"
            params.append(_validate_date(date_to))

    query += " ORDER BY date DESC, id DESC LIMIT ?"
    params.append(max(1, min(limit, 200)))

    with _conn_ready() as conn:
        rows = [dict(r) for r in conn.execute(query, params).fetchall()]

    if not rows:
        return "No expenses match those filters."

    total = sum(r["amount"] for r in rows)
    lines = [f"Expenses ({len(rows)}), total {_fmt_amount(total)}:"]
    lines.extend(_fmt_expense_row(r) for r in rows)
    return "\n".join(lines)


def _do_update(expense_id, amount, category, merchant, date, notes) -> str:
    if expense_id is None:
        raise ValueError("expense_id is required")

    with _conn_ready() as conn:
        existing = conn.execute("SELECT * FROM spending WHERE id = ?", (expense_id,)).fetchone()
        if existing is None:
            return f"Error: no expense found with id {expense_id}."

        updates: list[str] = []
        params: list = []
        changed: list[str] = []

        if amount is not None:
            a = _validate_amount(amount)
            updates.append("amount = ?")
            params.append(a)
            changed.append(f"amount → {_fmt_amount(a)}")

        if category is not None:
            c = _validate_category(category)
            updates.append("category = ?")
            params.append(c)
            changed.append(f"category → {c}")

        if merchant is not None:
            updates.append("merchant = ?")
            params.append(merchant or None)
            changed.append(f"merchant → {merchant or '(cleared)'}")

        if date is not None:
            d = _validate_date(date)
            updates.append("date = ?")
            params.append(d)
            changed.append(f"date → {d}")

        if notes is not None:
            updates.append("notes = ?")
            params.append(notes or None)
            changed.append("notes updated")

        if not updates:
            return f"No changes specified for expense #{expense_id}."

        params.append(expense_id)
        conn.execute(f"UPDATE spending SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()

    return f"Updated expense #{expense_id}: " + "; ".join(changed)


def _do_delete(expense_id) -> str:
    if expense_id is None:
        raise ValueError("expense_id is required")

    with _conn_ready() as conn:
        existing = conn.execute("SELECT * FROM spending WHERE id = ?", (expense_id,)).fetchone()
        if existing is None:
            return f"Error: no expense found with id {expense_id}."
        conn.execute("DELETE FROM spending WHERE id = ?", (expense_id,))
        conn.commit()

    return f"Deleted expense #{expense_id}: {_fmt_amount(existing['amount'])} ({existing['category']})"


def _do_budget(category, month, set_amount) -> str:
    month = month or _current_month()

    with _conn_ready() as conn:
        # Setting a budget
        if set_amount is not None:
            amount = _validate_amount(set_amount)
            cat_key = _validate_category(category) if category else None
            conn.execute(
                "INSERT INTO budgets (category, month, amount) VALUES (?, ?, ?) "
                "ON CONFLICT(category, month) DO UPDATE SET amount = excluded.amount",
                (cat_key, month, amount),
            )
            conn.commit()
            scope = f"'{cat_key}'" if cat_key else "overall"
            return f"Budget set: {scope} for {month} = {_fmt_amount(amount)}"

        # Checking a budget
        start, end = _month_bounds(month)
        cat_key = _validate_category(category) if category else None

        if cat_key:
            row = conn.execute(
                "SELECT amount FROM budgets WHERE category = ? AND month = ?",
                (cat_key, month),
            ).fetchone()
            spent = conn.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM spending "
                "WHERE category = ? AND date >= ? AND date <= ?",
                (cat_key, start, end),
            ).fetchone()[0]
            scope_label = cat_key
        else:
            row = conn.execute(
                "SELECT amount FROM budgets WHERE category IS NULL AND month = ?",
                (month,),
            ).fetchone()
            spent = conn.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM spending "
                "WHERE date >= ? AND date <= ?",
                (start, end),
            ).fetchone()[0]
            scope_label = "overall"

    if row is None:
        return (
            f"No budget set for {scope_label} in {month}. "
            f"You've spent {_fmt_amount(spent)} so far. "
            f"Set one with action='budget', set_amount=<value>."
        )

    budget_amount = row["amount"]
    pct = spent / budget_amount if budget_amount else 0
    remaining = budget_amount - spent

    lines = [
        f"Budget check: {scope_label} — {month}",
        f"  Budget:    {_fmt_amount(budget_amount)}",
        f"  Spent:     {_fmt_amount(spent)} ({pct * 100:.1f}%)",
        f"  {'Remaining' if remaining >= 0 else 'Over by'}: {_fmt_amount(abs(remaining))}",
        f"  [{_bar(pct)}]",
    ]
    if pct >= 1.0:
        lines.append("  ⚠ Budget exceeded.")
    elif pct >= 0.85:
        lines.append("  ⚠ Approaching budget limit.")

    return "\n".join(lines)


def _do_report(period) -> str:
    start, end, label = _resolve_period(period)

    with _conn_ready() as conn:
        rows = conn.execute(
            "SELECT category, SUM(amount) as total, COUNT(*) as n FROM spending "
            "WHERE date >= ? AND date <= ? GROUP BY category ORDER BY total DESC",
            (start, end),
        ).fetchall()
        grand_total = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM spending WHERE date >= ? AND date <= ?",
            (start, end),
        ).fetchone()[0]

    if not rows:
        return f"No expenses recorded for {label} ({start} to {end})."

    lines = [f"Spending report: {label} ({start} to {end})", f"Total: {_fmt_amount(grand_total)}", ""]
    for r in rows:
        pct = r["total"] / grand_total if grand_total else 0
        lines.append(
            f"  {r['category']:<14} {_fmt_amount(r['total']):>12}  "
            f"({pct * 100:5.1f}%, {r['n']} entries)  [{_bar(pct, 15)}]"
        )

    return "\n".join(lines)


# ── public async entrypoint ───────────────────────────────────────────────────

async def spending(
    action: str,
    expense_id: int | None = None,
    amount: float | None = None,
    category: str | None = None,
    merchant: str | None = None,
    date: str | None = None,
    notes: str | None = None,
    month: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    set_amount: float | None = None,
    period: str = "this_month",
    limit: int = 20,
) -> str:
    """
    Log and track personal spending. Manual logging only (Phase 1) — tell
    Kairos what you spent and it records it. No bank/UPI auto-sync yet.

    Args:
        action:      which operation to perform (see below). Required.
        expense_id:  target expense ID. Required for update/delete.
        amount:      expense amount (positive number). Required for log.
        category:    one of: food, transport, utilities, entertainment,
                     health, shopping, subscriptions, rent, education, other.
        merchant:    optional merchant/vendor name, e.g. "Swiggy", "BigBasket".
        date:        ISO date "YYYY-MM-DD". Defaults to today for log.
        notes:       optional free-text note, e.g. "team lunch".
        month:       "YYYY-MM" filter for list, or target month for budget.
                     Defaults to current month for budget.
        date_from:   filter for list — expenses on/after this date.
        date_to:     filter for list — expenses on/before this date.
        set_amount:  for action='budget' — set the budget instead of checking it.
        period:      for action='report' — "this_month", "last_month",
                     "last_7_days", "last_30_days", or explicit "YYYY-MM".
        limit:       max rows for list. Default 20, max 200.

    Actions:
        log     — record a new expense. Requires: amount, category.
        list    — list expenses, newest first. Filter by category/month/date range.
        update  — change fields on an existing expense. Requires: expense_id.
        delete  — remove a mistakenly logged expense. Requires: expense_id.
        budget  — check spend-to-date vs budget for a category (or overall) in
                  a given month. Pass set_amount to set/update the budget instead.
        report  — category breakdown with totals and percentages for a period.

    Returns a plain string in all cases. Never raises to the caller.
    """
    action = (action or "").strip().lower()
    valid_actions = {"log", "list", "update", "delete", "budget", "report"}
    if action not in valid_actions:
        return f"Error: Unknown action '{action}'. Valid: {', '.join(sorted(valid_actions))}."

    try:
        if action == "log":
            return await asyncio.to_thread(_do_log, amount, category, merchant, date, notes)

        if action == "list":
            return await asyncio.to_thread(_do_list, category, month, date_from, date_to, limit)

        if action == "update":
            return await asyncio.to_thread(
                _do_update, expense_id, amount, category, merchant, date, notes
            )

        if action == "delete":
            return await asyncio.to_thread(_do_delete, expense_id)

        if action == "budget":
            return await asyncio.to_thread(_do_budget, category, month, set_amount)

        if action == "report":
            return await asyncio.to_thread(_do_report, period)

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.exception("spending() failed for action=%r", action)
        return f"Error: Spending tool failed unexpectedly — {type(e).__name__}: {e}"

    return "Error: Unhandled action path."