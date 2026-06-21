# spending

**← [Back to Tool Guide](../TOOL_GUIDE.md)**

**File:** `runtime/tools/spending.py`
**Env vars required:** none — uses the local `kairos.db`
**Execution pattern:** agentic
**Added:** Mark 3

Manual expense logging, budgets, and category reports. Reuses the `spending` table from `memory/sqlite_store.py`, adding two columns via a guarded, idempotent `ALTER TABLE` (`source`, `sms_id`) so a future Phase 2 SMS-auto-logging feature can land without a migration.

**Phase 1 scope:** manual logging only. No UPI/bank API integration — the user tells Kairos what they spent.

## When it's used

- User says they spent money on something, or asks about spending/budget
- Classifier sets `domains: [spending]` and `tools_needed: [spending]`

## Parameters

| Parameter      | Type    | Required for                       | Description |
|-----------------|---------|---------------------------------------|--------------|
| `action`        | string  | all                                     | Which operation to perform |
| `expense_id`    | integer | `update`, `delete`                       | Target expense ID |
| `amount`        | number  | `log`                                     | Positive number, required |
| `category`      | string  | `log`                                      | One of the fixed categories below |
| `merchant`      | string  | optional                                    | e.g. `"Swiggy"`, `"BigBasket"` |
| `date`          | string  | optional                                     | ISO `YYYY-MM-DD`. Defaults to today for `log` |
| `notes`         | string  | optional                                      | Free text, e.g. `"team lunch"` |
| `month`         | string  | optional, `list` filter / `budget` target     | `"YYYY-MM"`. Defaults to current month for `budget` |
| `date_from`     | string  | optional, `list` filter                        | On or after this ISO date |
| `date_to`       | string  | optional, `list` filter                         | On or before this ISO date |
| `set_amount`    | number  | for `budget` — set instead of check               | Sets/updates the budget amount |
| `period`        | string  | `report`                                            | `this_month` (default) \| `last_month` \| `last_7_days` \| `last_30_days` \| explicit `"YYYY-MM"` |
| `limit`         | integer | optional, `list`                                      | Default 20, max 200 |

**Valid categories:** `food`, `transport`, `utilities`, `entertainment`, `health`, `shopping`, `subscriptions`, `rent`, `education`, `other`

## Actions

| Action     | What it does | Required params |
|-------------|---------------|--------------------|
| `log`       | Record a new expense | `amount`, `category` |
| `list`      | List expenses newest-first, with a running total. Filterable by `category`/`month`/`date_from`/`date_to` | — |
| `update`    | Change `amount`, `category`, `merchant`, `date`, or `notes` | `expense_id` + ≥1 field |
| `delete`    | Remove a mistakenly logged expense | `expense_id` |
| `budget`    | Check spend-to-date vs. budget for a category (or overall) in a month. Pass `set_amount` to set/update the budget instead of checking | — (set: `set_amount`) |
| `report`    | Category breakdown with totals and percentages for a period, sorted highest-spend first | — |

## Example LLM calls

**Log an expense:**
```json
{ "name": "spending", "input": { "action": "log", "amount": 500, "category": "food", "merchant": "BigBasket" } }
```

**Set a monthly budget for a category:**
```json
{ "name": "spending", "input": { "action": "budget", "category": "food", "month": "2026-06", "set_amount": 5000 } }
```

**Check that budget:**
```json
{ "name": "spending", "input": { "action": "budget", "category": "food", "month": "2026-06" } }
```

**This month's report:**
```json
{ "name": "spending", "input": { "action": "report", "period": "this_month" } }
```

## Return format (budget check)

```
Budget check: food — 2026-06
  Budget:    ₹5,000.00
  Spent:     ₹4,300.00 (86.0%)
  Remaining: ₹700.00
  [█████████████████░░░]
  ⚠ Approaching budget limit.
```

## Return format (report)

```
Spending report: 2026-06 (2026-06-01 to 2026-06-30)
Total: ₹10,000.00

  rent           ₹5,000.00  (50.0%, 1 entries)  [███████████████]
  food           ₹3,000.00  (30.0%, 3 entries)  [█████████░░░░░░]
  transport      ₹2,000.00  (20.0%, 2 entries)  [██████░░░░░░░░░]
```

## Failure modes

| Condition                                  | Return value |
|-----------------------------------------------|-----------------|
| Unknown action                                  | `Error: Unknown action '<action>'. Valid: budget, delete, list, log, report, update.` |
| `log` missing `amount` or `category`             | `Error: amount must be a number, got None` / `Error: category is required. Valid: <list>` |
| `category` not in the fixed set                    | `Error: Unknown category '<value>'. Valid: <list>` |
| `amount` zero/negative/non-numeric                   | `Error: amount must be positive, got <amount>` |
| Invalid date format                                     | `Error: date must be YYYY-MM-DD, got '<value>'` |
| `update`/`delete` on nonexistent `expense_id`             | `Error: no expense found with id <id>.` |
| `update` called with no fields                               | `No changes specified for expense #<id>.` |
| No budget set yet when checking                                | `No budget set for <scope> in <month>. You've spent ₹<amount> so far. Set one with action='budget', set_amount=<value>.` |
| `report` with no expenses in the period                           | `No expenses recorded for <label> (<start> to <end>).` |
| Unrecognized `period`                                                | `Error: Unrecognised period '<period>'. Use one of (this_month, last_month, last_7_days, last_30_days) or an explicit 'YYYY-MM'.` |
| Unexpected exception                                                    | `Error: Spending tool failed unexpectedly — <ExceptionType>: <detail>` |

## Notes

- Budgets are keyed by `(category, month)`; an overall (no-category) budget for a month uses `NULL` category and aggregates all spending regardless of category.
- The budget bar warns at ≥85% spent ("Approaching budget limit") and ≥100% ("Budget exceeded").
- All currency is formatted as `₹X,XXX.XX` — there is currently no multi-currency support.
