# habits

**← [Back to Tool Guide](../TOOL_GUIDE.md)**

**File:** `runtime/tools/habits.py`
**Env vars required:** none — uses the local `kairos.db`
**Execution pattern:** agentic
**Added:** Mark 3

Habit creation, checkins, streaks, and a 30-day consistency heatmap. Reuses the `habits` table from `memory/sqlite_store.py` (so the orchestrator's existing context block keeps working unchanged) and adds one companion table, `habit_logs` — one row per `(habit_id, date)` checkin, the source of truth for all streak/consistency math. `habits.streak` and `habits.last_done` are kept as a denormalized cache derived from the log, recomputed on every checkin/undo.

A bare running counter can desync from reality if a checkin is backfilled, undone, or the app crashes mid-update — `habit_logs` makes that impossible.

## When it's used

- User wants to track a new habit, check one off, or see their streak
- Classifier sets `domains: [habits]` and `tools_needed: [habits]`

## Parameters

| Parameter           | Type    | Required for                              | Description |
|----------------------|---------|---------------------------------------------|--------------|
| `action`             | string  | all                                          | Which operation to perform |
| `habit_id`           | integer | `checkin`, `undo_checkin`, `stats`, `delete`  | Target habit ID |
| `name`               | string  | `create`                                      | e.g. `"meditation"`, `"gym"`, `"DSA practice"` |
| `target_frequency`   | string  | optional, `create`                              | One of the fixed values below. Defaults to `daily` |
| `date`               | string  | optional, `checkin`/`undo_checkin`               | ISO `YYYY-MM-DD`. Defaults to today; pass a past date to backfill |

**Valid `target_frequency` values:** `daily`, `weekdays`, `3x_week`, `5x_week`, `weekly`

## Actions

| Action          | What it does | Required params |
|------------------|---------------|--------------------|
| `create`         | Add a new habit. Rejects duplicate names | `name` |
| `list`           | All habits with current streak, last checkin date, and a done-today flag | — |
| `checkin`        | Log a completion for today (or a backfilled past date). Recomputes streak from the full log | `habit_id` |
| `undo_checkin`    | Remove a checkin to fix a mistake; recomputes streak | `habit_id` |
| `stats`            | Streak, 30-day consistency %, and an ASCII heatmap for one habit | `habit_id` |
| `delete`             | Permanently remove a habit and its full checkin history | `habit_id` |

## Example LLM calls

**Create:**
```json
{ "name": "habits", "input": { "action": "create", "name": "DSA practice", "target_frequency": "daily" } }
```

**Check in for today:**
```json
{ "name": "habits", "input": { "action": "checkin", "habit_id": 3 } }
```

**Backfill a missed day:**
```json
{ "name": "habits", "input": { "action": "checkin", "habit_id": 3, "date": "2026-06-18" } }
```

**View stats:**
```json
{ "name": "habits", "input": { "action": "stats", "habit_id": 3 } }
```

## Return format (list)

```
Habits (2):
  #1    Meditation               streak:   5  target: daily      last: 2026-06-21 ✓ done today
  #2    Gym                      streak:   0  target: 3x_week    last: never ⚠ not done today
```

## Return format (stats)

```
Stats for 'Meditation' (#1):
  Target:           daily
  Current streak:   5
  Last checkin:     2026-06-21
  Total checkins:   18
  30-day consistency: 60%

  Last 30 days: ······■■■■■···■■■■■■■■■■■■■■■
  (oldest → newest, ■ = done, · = missed)
```

## Failure modes

| Condition                                | Return value |
|---------------------------------------------|-----------------|
| Unknown action                                | `Error: Unknown action '<action>'. Valid: checkin, create, delete, list, stats, undo_checkin.` |
| `create` with empty `name`                      | `Error: name must not be empty` |
| `create` with duplicate `name`                    | `Error: a habit named '<name>' already exists (#<id>).` |
| Invalid `target_frequency`                          | `Error: target_frequency must be one of ('daily', 'weekdays', '3x_week', '5x_week', 'weekly') — got '<value>'` |
| Invalid `date` format                                 | `Error: date must be YYYY-MM-DD, got '<value>'` |
| `checkin`/`undo_checkin`/`stats`/`delete` on nonexistent `habit_id` | `Error: no habit found with id <id>.` |
| Duplicate checkin on the same date                       | `Habit '<name>' already checked in for <date>.` |
| `undo_checkin` with nothing to remove                       | `No checkin found for '<name>' on <date>.` |
| Unexpected exception                                          | `Error: Habits tool failed unexpectedly — <ExceptionType>: <detail>` |

## Notes

- A streak isn't considered "broken" until the day is fully over — if today has no checkin yet, the streak still counts through yesterday rather than resetting to 0 mid-day.
- `weekdays` frequency skips Saturday/Sunday entirely in both streak counting and the 30-day expected-occurrence math used for consistency %.
- For `3x_week`/`5x_week`/`weekly`, "streak" means consecutive ISO weeks (Mon–Sun) meeting the target — not consecutive days.
