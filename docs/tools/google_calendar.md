# google_calendar

**← [Back to Tool Guide](../TOOL_GUIDE.md)**

**File:** `runtime/tools/google_calendar.py`
**Env vars required:** none directly — uses OAuth `token.json` / `credentials.json` in the project root (`GOOGLE_CALENDAR_CREDENTIALS_PATH` overrides the credentials path, default `credentials.json`)
**Scope:** `https://www.googleapis.com/auth/calendar` (read + write)
**Execution pattern:** agentic

## When it's used

- User asks to view, create, update, or delete calendar events
- Classifier sets `domains: [events]` and `tools_needed: [google_calendar]`

## Parameters

| Parameter         | Type    | Required for                                    | Description |
|--------------------|---------|--------------------------------------------------|--------------|
| `action`           | string  | all                                              | Which operation to perform (see actions table) |
| `max_results`      | integer | `list_events`, `search_events`                   | 1–50, default 10 |
| `query`            | string  | `search_events`; optional for `delete_event`/`update_event` when `event_id` is unknown | Free-text keyword match |
| `event_id`         | string  | `get_event`; preferred over `query` for `delete_event`/`update_event` | Get IDs from `list_events`/`search_events` output |
| `calendar_id`      | string  | optional, all actions                            | Defaults to `GOOGLE_CALENDAR_ID` env var or `"primary"`. Get IDs from `list_calendars` |
| `summary`          | string  | `create_event`                                   | Event title |
| `description`      | string  | optional, `create_event`                         | Event notes/body |
| `location`         | string  | optional, `create_event`                         | Location string |
| `start_time`       | string  | `create_event`                                   | ISO 8601, e.g. `2026-06-20T15:00:00` (no offset needed — timezone applied from `TIMEZONE` env var) |
| `end_time`         | string  | `create_event`                                   | ISO 8601 end datetime |
| `attendee_emails`  | array of strings | optional, `create_event`                  | Invite emails — sent automatically when provided |
| `new_summary`      | string  | optional, `update_event`                         | Replacement title |
| `new_description`  | string  | optional, `update_event`                         | Replacement description |
| `new_start_time`   | string  | optional, `update_event`                         | Replacement start datetime |
| `new_end_time`     | string  | optional, `update_event`                         | Replacement end datetime |
| `new_location`     | string  | optional, `update_event`                         | Replacement location |

## Actions

| Action            | What it does | Key parameters |
|--------------------|--------------|-----------------|
| `list_events`      | Upcoming events from now, soonest first | `max_results`, `calendar_id` |
| `get_event`        | Full details of one event — notes, attendees, Meet link | `event_id` |
| `search_events`    | Free-text search across all events, past and future | `query`, `max_results` |
| `create_event`     | Create a new event, returns the new event ID | `summary`, `start_time`, `end_time` |
| `update_event`     | Patch only the fields provided (uses `events().patch()`, never a full replace) | `event_id` or `query`, plus `new_*` fields |
| `delete_event`     | Delete by `event_id`, or by `query` search (deletes first match if ambiguous, lists the rest) | `event_id` or `query` |
| `list_calendars`   | List all calendars the user has access to, with IDs and access role | — |

## Example LLM calls

**List upcoming events:**
```json
{ "name": "google_calendar", "input": { "action": "list_events", "max_results": 5 } }
```

**Create an event with attendees:**
```json
{
  "name": "google_calendar",
  "input": {
    "action": "create_event",
    "summary": "Team standup",
    "start_time": "2026-06-20T15:00:00",
    "end_time": "2026-06-20T15:30:00",
    "description": "Weekly sync with the ML team",
    "attendee_emails": ["alice@example.com"]
  }
}
```

**Delete by name search:**
```json
{ "name": "google_calendar", "input": { "action": "delete_event", "query": "dentist appointment" } }
```

**Update event timing:**
```json
{
  "name": "google_calendar",
  "input": {
    "action": "update_event",
    "query": "team standup",
    "new_start_time": "2026-06-20T16:00:00",
    "new_end_time": "2026-06-20T16:30:00"
  }
}
```

**List available calendars:**
```json
{ "name": "google_calendar", "input": { "action": "list_calendars" } }
```

## Return format (list_events)

```
Upcoming events (2 shown):
ID: abc123xyz | Fri 20 Jun 2026, 3:00 PM → Fri 20 Jun 2026, 3:30 PM | Team standup
  Location: Conference Room A

ID: def456uvw | Sat 21 Jun 2026, 10:00 AM → Sat 21 Jun 2026, 11:00 AM | Doctor appointment
```

The `ID:` prefix exposes the event ID for follow-up `get_event`, `update_event`, or `delete_event` calls without a search round-trip.

## Failure modes

| Condition                                   | Return value                                                                 |
|-----------------------------------------------|---------------------------------------------------------------------------------|
| `credentials.json` missing                    | `Setup error: Missing credentials file at '<path>'. Download it from Google Cloud Console and place it in the project root.` |
| `event_id` not found                          | `Event not found: ID '<id>' does not exist on this calendar.` (404), or `... — nothing deleted.` for delete |
| Event already deleted (delete_event)          | `Event '<id>' was already deleted.` (410)                                       |
| `search_events` with empty `query`            | `Error: 'search_events' requires a non-empty 'query'.`                          |
| `get_event` without `event_id`                | `Error: 'get_event' requires an 'event_id'. Get IDs from list_events or search_events.` |
| `create_event` missing required field(s)      | `Error: 'create_event' requires: <missing fields>.`                             |
| `delete_event`/`update_event` with neither `event_id` nor `query` | `Error: 'delete_event' requires either 'event_id' or 'query'.` (same pattern for update) |
| `update_event` called with no `new_*` fields  | `Nothing to update — no new fields were provided.`                              |
| Unknown action                                | `Error: unknown action '<action>'. Valid actions: list_events, get_event, search_events, create_event, update_event, delete_event, list_calendars.` |
| Network/API error                             | `Error <verb>ing event(s): <detail>` / `Error executing calendar action '<action>': <detail>` |

## First-time OAuth setup

```bash
# credentials.json must exist in the project root first (from Google Cloud Console)
cd runtime
python -c "from tools.google_calendar import _get_service; _get_service()"
# Opens a browser for Google sign-in; token.json is written after success
```
