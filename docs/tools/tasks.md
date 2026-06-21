# tasks

**← [Back to Tool Guide](../TOOL_GUIDE.md)**

**File:** `runtime/tools/tasks.py`
**Env vars required:** none — uses the local `kairos.db` (`DATA_DIR` env var controls location)
**Execution pattern:** agentic
**Added:** Mark 3

The agentic layer for to-dos. Reuses the same `tasks` table that `memory/sqlite_store.py` creates and that the orchestrator's context assembler already reads from (`fetch_open_tasks()`). This tool is what the LLM calls directly to create/update/search/complete tasks mid-conversation — `sqlite_store.py` is unaffected and still feeds passive context injection.

**Phase 1 scope:** tasks are intentionally separate from Google Calendar — no `calendar_event_id` linking yet. Use `tasks` for to-dos/work items, [google_calendar](google_calendar.md) for meetings and scheduled events.

## When it's used

- User wants to add, view, update, complete, or search a to-do
- Classifier sets `domains: [tasks]` and `tools_needed: [tasks]`

## Parameters

| Parameter     | Type    | Required for                              | Description |
|----------------|---------|---------------------------------------------|--------------|
| `action`       | string  | all                                          | Which operation to perform |
| `task_id`      | integer | `update`, `complete`, `delete`               | Target task ID |
| `title`        | string  | `create`; optional for `update`              | Task title |
| `due_date`     | string  | optional                                      | ISO `YYYY-MM-DD`. For `update`, pass `""` to explicitly clear an existing due date |
| `project`      | string  | optional                                      | Free-text project/category tag, e.g. `"kairos"`, `"college"` |
| `priority`     | integer | optional                                      | `1`=low, `2`=normal (default), `3`=high |
| `status`       | string  | optional, `list` filter                       | `"open"` or `"done"`. Default for `list` is open-only |
| `due_before`   | string  | optional, `list` filter                       | ISO date — tasks due on or before |
| `due_after`    | string  | optional, `list` filter                       | ISO date — tasks due on or after |
| `query`        | string  | `search`                                       | Search text |
| `undo`         | boolean | optional, `complete`                            | Pass `true` to reopen a completed task instead of closing it |
| `days`         | integer | optional, `stats`                                | Lookback window. Default 7 |
| `limit`        | integer | optional, `list`/`search`                          | Default 20, max 100 |

## Actions

| Action       | What it does | Required params |
|---------------|---------------|--------------------|
| `create`      | Add a new task | `title` |
| `list`        | List tasks, defaults to open tasks sorted by priority desc then due date asc. Filterable by `status`/`priority`/`due_before`/`due_after`/`project` | — |
| `update`      | Change `title`, `due_date`, `project`, or `priority` | `task_id` + ≥1 field |
| `complete`    | Mark done; pass `undo=true` to reopen | `task_id` |
| `delete`      | Permanently remove | `task_id` |
| `search`      | Full-text search over title + project | `query` |
| `stats`       | Counts by status/priority, overdue count, due-today count | — |

## Example LLM calls

**Create with due date and priority:**
```json
{ "name": "tasks", "input": { "action": "create", "title": "Finish API integration", "due_date": "2026-06-27", "priority": 3 } }
```

**List overdue + due-soon tasks:**
```json
{ "name": "tasks", "input": { "action": "list", "due_before": "2026-06-25" } }
```

**Complete a task:**
```json
{ "name": "tasks", "input": { "action": "complete", "task_id": 12 } }
```

**Clear a due date:**
```json
{ "name": "tasks", "input": { "action": "update", "task_id": 12, "due_date": "" } }
```

## Return format (list)

```
Tasks (3):
  #12   (high  ) [kairos] Finish API integration  — due TODAY
  #9    (normal)         Write documentation
  #5    (low   )         Read changelog  — OVERDUE (was due 2026-06-10)
```

## Return format (stats)

```
Task stats (last 7 day window requested):
  Open:        4
  Done:        9
  Overdue:     1
  Due today:   1
  Open by priority:
    high  : 1
    normal: 2
    low   : 1
```

## Failure modes

| Condition                              | Return value |
|---------------------------------------------|-----------------|
| Unknown action                                  | `Error: Unknown action '<action>'. Valid: complete, create, delete, list, search, stats, update.` |
| `create` with empty title                        | `Error: title must not be empty` |
| Invalid `priority` (not 1/2/3)                     | `Error: priority must be 1 (low), 2 (normal), or 3 (high) — got <n>` |
| Invalid date format                                  | `Error: due_date must be YYYY-MM-DD, got '<value>'` |
| `update`/`complete`/`delete` on nonexistent `task_id`  | `Error: no task found with id <id>.` |
| `update` called with no fields                           | `No changes specified for task #<id>.` |
| `complete` on an already-done task                          | `Task #<id> ("<title>") is already marked done.` (same pattern for undo on an already-open task) |
| `search` with empty query                                      | `Error: query must not be empty` |
| Unexpected exception                                              | `Error: Tasks tool failed unexpectedly — <ExceptionType>: <detail>` |

## Notes

- All blocking SQLite calls are wrapped in `asyncio.to_thread()` — this fixes a latent bug in `sqlite_store.py`, where the equivalent functions are plain sync and would raise `TypeError` if awaited directly.
- `list` results are sorted by priority desc, then nulls-last due date asc — tasks with no due date sort after dated ones at the same priority.
