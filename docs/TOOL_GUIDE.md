# Kairos Tool Guide

> Complete reference for every tool in the registry — parameters, actions, examples, and failure modes.

Tools are invoked by the LLM during the agentic tool loop. The classifier flags `tools_needed`; the executor validates inputs against the schema, then dispatches. Every tool returns a plain string — never raises to the LLM.

---

## Contents

- [web\_search](#web_search)
- [send\_message](#send_message)
- [google\_calendar](#google_calendar)
- [check\_gmail](#check_gmail)
- [Adding a new tool](#adding-a-new-tool)

---

## web\_search

**File:** `runtime/tools/web_search.py`  
**Env vars required:** none (DuckDuckGo default). Optional: `BRAVE_API_KEY`, `TAVILY_API_KEY`, `SERPER_API_KEY`  
**Backend:** set via `SEARCH_BACKEND` in `.env` — `duckduckgo` (default) | `brave` | `tavily` | `serper`

### When the LLM uses this

- User asks about current events, news, prices, or anything that may have changed recently
- User asks a factual question the model cannot confidently answer from memory
- Classifier sets `intent: search` or `needs_external_tools: true`

### Parameters

| Parameter | Type   | Required | Constraints          | Description                     |
|-----------|--------|----------|----------------------|---------------------------------|
| `query`   | string | yes      | 1–200 chars          | The search query to send        |

### Example LLM call

```json
{
  "name": "web_search",
  "input": {
    "query": "India vs Australia Test series 2026 results"
  }
}
```

### Return format

```
Search results for 'India vs Australia Test series 2026 results':

- India win third Test to take series lead
  Australia's batting collapsed on day four as India claimed victory by an innings.
  https://example-cricket-site.com/...

- Australia's captain responds to defeat
  ...
  https://...
```

### Failure modes

| Condition                         | Return value                          |
|-----------------------------------|---------------------------------------|
| No results found                  | `No results found for: <query>`       |
| Network error / backend down      | `Search results for '...': (empty)`   |
| `BRAVE_API_KEY` not set           | Falls back to DuckDuckGo silently     |
| Unknown `SEARCH_BACKEND` value    | Falls back to DuckDuckGo with warning |

### Config

```env
SEARCH_BACKEND=duckduckgo      # no key needed
SEARCH_BACKEND=brave           # needs BRAVE_API_KEY
SEARCH_BACKEND=tavily          # needs TAVILY_API_KEY
SEARCH_BACKEND=serper          # needs SERPER_API_KEY
SEARCH_MAX_RESULTS=5           # default: 5
```

---

## send\_message

**File:** `runtime/tools/messaging.py`  
**Env vars required:** `TELEGRAM_BOT_TOKEN`, `TELEGRAM_USER_ID`

### When the LLM uses this

- User explicitly asks Kairos to send them something on Telegram
- Kairos wants to push a proactive alert (reminder, briefing chunk, etc.)
- Classifier sets `tools_needed: [send_message]`

### Parameters

| Parameter | Type   | Required | Constraints          | Description                          |
|-----------|--------|----------|----------------------|--------------------------------------|
| `message` | string | yes      | max 4096 chars       | The message text to send to the user |

> **4096 char hard limit** — Telegram's API rejects longer messages. If the content is longer, the tool itself returns an error string instructing the LLM to chunk it and call again with `Part 1 of N` framing.

### Example LLM call

```json
{
  "name": "send_message",
  "input": {
    "message": "Reminder: your 3pm standup starts in 10 minutes."
  }
}
```

### Return format

```
SUCCESS: Message sent to Telegram. You must now reply to the user in text confirming it was sent, and DO NOT call this tool again.
```

The success string contains an explicit directive preventing the LLM from calling `send_message` again on the same turn (local models tend to loop).

### Failure modes

| Condition                          | Return value                                      |
|------------------------------------|---------------------------------------------------|
| Message exceeds 4096 chars         | `Error: Message too long. Telegram max is 4096 characters. Your message was N characters...` |
| `TELEGRAM_BOT_TOKEN` not set       | `Error: TELEGRAM_BOT_TOKEN or TELEGRAM_USER_ID is missing...` |
| Telegram API network error         | `Error sending message to Telegram API: <detail>` |

### Notes

- Supports emoji, newlines, and standard Telegram text formatting
- Does not support inline keyboards, photos, or files — text only
- The tool is intentionally simple; the spec doc covers why the event-bus approach was over-engineering

---

## google\_calendar

**File:** `runtime/tools/google_calendar.py`  
**Env vars required:** none in `.env` — uses `token.json` / `credentials.json` in the project root (Google OAuth flow)  
**Scope:** `https://www.googleapis.com/auth/calendar` (read + write)

### When the LLM uses this

- User asks to view, create, update, or delete calendar events
- Classifier sets `domains: [events]` and `tools_needed: [google_calendar]`

### Parameters

| Parameter        | Type    | Required for                          | Description                                                   |
|------------------|---------|---------------------------------------|---------------------------------------------------------------|
| `action`         | string  | all                                   | Which operation to perform (see actions table below)          |
| `max_results`    | integer | `list_events`                         | Max events to return. Default: 10                             |
| `query`          | string  | `search_events`, optional for `delete_event` / `update_event` | Free-text keyword to match against events    |
| `event_id`       | string  | optional for `delete_event` / `update_event` | Exact Google Calendar event ID. Preferred over `query` when available |
| `summary`        | string  | `create_event`                        | Title/name of the event                                       |
| `description`    | string  | optional for `create_event`           | Body/notes for the event                                      |
| `start_time`     | string  | `create_event`                        | ISO 8601 start datetime, e.g. `2026-06-20T14:00:00`          |
| `end_time`       | string  | `create_event`                        | ISO 8601 end datetime                                         |
| `new_summary`    | string  | optional for `update_event`           | New title to replace the existing one                         |
| `new_description`| string  | optional for `update_event`           | New description to replace the existing one                   |
| `new_start_time` | string  | optional for `update_event`           | New start datetime                                            |
| `new_end_time`   | string  | optional for `update_event`           | New end datetime                                              |

### Actions

| Action          | What it does                                                         | Key parameters            |
|-----------------|----------------------------------------------------------------------|---------------------------|
| `list_events`   | Lists upcoming events from now, soonest first                        | `max_results`             |
| `search_events` | Free-text search across all events (past and future)                 | `query`, `max_results`    |
| `create_event`  | Creates a new event on the primary calendar                          | `summary`, `start_time`, `end_time` |
| `delete_event`  | Deletes an event by `event_id` or by searching with `query`          | `event_id` or `query`     |
| `update_event`  | Updates fields of an existing event by `event_id` or `query` search  | `event_id` or `query`, then `new_*` fields |

### Example LLM calls

**List upcoming events:**
```json
{
  "name": "google_calendar",
  "input": {
    "action": "list_events",
    "max_results": 5
  }
}
```

**Create an event:**
```json
{
  "name": "google_calendar",
  "input": {
    "action": "create_event",
    "summary": "Team standup",
    "start_time": "2026-06-20T15:00:00",
    "end_time": "2026-06-20T15:30:00",
    "description": "Weekly sync with the ML team"
  }
}
```

**Delete by name search:**
```json
{
  "name": "google_calendar",
  "input": {
    "action": "delete_event",
    "query": "dentist appointment"
  }
}
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

### Return format (list_events)

```
Upcoming Calendar Events:
- ID: abc123xyz | [2026-06-20T15:00:00+05:30] Team standup
- ID: def456uvw | [2026-06-21T10:00:00+05:30] Doctor appointment
```

The `ID:` prefix exposes the event ID so the LLM can use it in a follow-up `delete_event` or `update_event` call instead of doing a keyword search.

### Failure modes

| Condition                          | Return value                                         |
|------------------------------------|------------------------------------------------------|
| `credentials.json` missing         | `Error: Missing 'credentials.json' in root directory.` |
| Event not found by query           | `Error: Could not find an event matching '...'`      |
| Network / API error                | `Error executing calendar action '...': <detail>`    |
| Missing required param for action  | `Error: ...` (specific message per action)           |

### First-time OAuth setup

```bash
# Run once — opens browser for Google sign-in
# credentials.json must exist in project root first (from Google Cloud Console)
cd runtime
python -c "from tools.google_calendar import _get_calendar_service; _get_calendar_service()"
# token.json is written to project root after successful auth
```

---

## check\_gmail

**File:** `runtime/tools/gmail_check.py`  
**Env vars required:** `GMAIL_USER`, `GMAIL_APP_PASSWORD`

> `GMAIL_APP_PASSWORD` is a 16-character Google App Password — not your Google account password. Generate one at: myaccount.google.com/apppasswords. IMAP must be enabled in Gmail settings (Settings → See all settings → Forwarding and POP/IMAP → Enable IMAP).

### When the LLM uses this

- User asks to check email, look for alerts, or summarise their inbox
- User asks about a specific email or wants to search by keyword
- Classifier sets `tools_needed: [check_gmail]`

### Parameters

| Parameter     | Type    | Required for                       | Constraints  | Description                                                                 |
|---------------|---------|------------------------------------|--------------|-----------------------------------------------------------------------------|
| `action`      | string  | all                                | enum         | Which operation to perform (see actions table)                              |
| `max_results` | integer | `list_unread`, `list_recent`, `search` | 1–20, default 5 | Max emails to return                                               |
| `query`       | string  | `search`                           | max 200 chars | Keyword to search for in subject/body/headers                             |
| `uid`         | string  | `get_body`, `mark_read`            | from prior output | Email UID from a previous list or search result                       |

### Actions

| Action         | What it does                                                    | Required params     | Network cost |
|----------------|-----------------------------------------------------------------|---------------------|--------------|
| `list_unread`  | Headers of unread emails, most recent first (default)           | —                   | Low          |
| `list_recent`  | Headers of recent emails, read or unread, most recent first     | —                   | Low          |
| `search`       | Search inbox by keyword — matches subject, body, and headers    | `query`             | Low          |
| `get_body`     | Full readable text body of one specific email                   | `uid`               | Medium       |
| `mark_read`    | Mark one email as read (sets the \\Seen IMAP flag)              | `uid`               | Low          |
| `count_unread` | Return unread count only — fastest option, no header data       | —                   | Minimal      |

### Example LLM calls

**Check unread (default — most common):**
```json
{
  "name": "check_gmail",
  "input": {
    "action": "list_unread",
    "max_results": 5
  }
}
```

**Search for a specific email:**
```json
{
  "name": "check_gmail",
  "input": {
    "action": "search",
    "query": "GitHub deployment failed",
    "max_results": 3
  }
}
```

**Read the full body of an email (UID from prior list/search output):**
```json
{
  "name": "check_gmail",
  "input": {
    "action": "get_body",
    "uid": "18943"
  }
}
```

**Mark an email as read:**
```json
{
  "name": "check_gmail",
  "input": {
    "action": "mark_read",
    "uid": "18943"
  }
}
```

**Just the count — fastest:**
```json
{
  "name": "check_gmail",
  "input": {
    "action": "count_unread"
  }
}
```

### Return format (list_unread)

```
Unread emails (showing 3 of 7):

1. UID: 18943
   FROM: GitHub <noreply@github.com>
   SUBJECT: [kairos] Deployment failed
   DATE: Thu, 18 Jun 2026 09:14:22 +0530

2. UID: 18940
   FROM: Google <no-reply@accounts.google.com>
   SUBJECT: Security alert
   DATE: Wed, 17 Jun 2026 22:01:45 +0530

3. UID: 18937
   FROM: Anthropic <billing@anthropic.com>
   SUBJECT: Your invoice is ready
   DATE: Wed, 17 Jun 2026 11:00:00 +0000
```

The `UID:` value is what you pass to `get_body` or `mark_read`.

### Return format (get\_body)

```
FROM: GitHub <noreply@github.com>
SUBJECT: [kairos] Deployment failed
DATE: Thu, 18 Jun 2026 09:14:22 +0530

BODY:
The workflow run "Deploy to prod" triggered by push to main failed.

Job: build-and-deploy
Step: Run pytest
Exit code: 1

View full details: https://github.com/...

[… truncated at 1500 chars]
```

### Failure modes

| Condition                          | Return value                                                         |
|------------------------------------|----------------------------------------------------------------------|
| `GMAIL_USER` or `GMAIL_APP_PASSWORD` not set | `Error: Gmail credentials missing or IMAP login failed.` |
| Wrong app password / IMAP disabled | `Error: Gmail credentials missing or IMAP login failed.`             |
| UID doesn't exist                  | `Could not fetch email with UID N. It may no longer exist.`          |
| `search` with empty `query`        | `Error: 'search' action requires a non-empty 'query' parameter.`     |
| `get_body` without `uid`           | `Error: 'get_body' action requires a 'uid' parameter. Get UIDs from list_unread or search.` |
| Network/IMAP error mid-operation   | `Failed to retrieve emails: <detail>`                                |
| Unknown action value               | `Error: unknown action '...'. Valid actions: ...`                    |

### Typical multi-turn pattern

```
User: "Any important emails?"
  → LLM calls check_gmail(action="list_unread", max_results=5)
  ← Returns 5 headers with UIDs

LLM summarises: "You have 7 unread. The GitHub deployment failure
  and the Anthropic invoice look important."

User: "Read the GitHub one"
  → LLM calls check_gmail(action="get_body", uid="18943")
  ← Returns full body text

LLM summarises the body content.

User: "Mark it read"
  → LLM calls check_gmail(action="mark_read", uid="18943")
  ← "Marked email UID 18943 as read."
```

---

## Adding a new tool

Full walkthrough in [CONTRIBUTING.md](CONTRIBUTING.md). Quick checklist:

1. **Write the handler** in `runtime/tools/your_tool.py`
   - Async function that accepts `**kwargs` matching your schema properties
   - Must never raise — return error strings on failure
   - All blocking I/O via `asyncio.to_thread()`

2. **Add a lazy loader** in `registry.py`
   ```python
   def _load_your_tool():
       from tools.your_tool import your_function
       return your_function
   ```

3. **Register in `REGISTRY`**
   ```python
   "your_tool": {
       "description": "...",        # shown to the LLM — be specific
       "schema": { ... },           # JSON schema, validated before dispatch
       "handler": _load_your_tool,
       "enabled": True,
       "requires_env": ["YOUR_API_KEY"],
   }
   ```

4. **Add to `VALID_TOOLS`** in `classifier.py` so the classifier can flag it

5. **Test with executor directly**
   ```python
   from tools.executor import execute
   import asyncio
   result = asyncio.run(execute("your_tool", {"param": "value"}))
   print(result)
   ```

### Tool contract rules

| Rule                              | Why                                                                  |
|-----------------------------------|----------------------------------------------------------------------|
| Never raise from a tool           | LLM receives the return string; an exception breaks the tool loop    |
| Return strings only               | Executor casts to `str(result)` — return rich text, not dicts        |
| Validate inside the handler too   | Schema catches shape errors; handler catches semantic errors         |
| Keep handler focused              | One tool, one concern — `check_gmail` doesn't also send messages     |
| Use `asyncio.to_thread()` for blocking I/O | Never block the event loop with IMAP, file reads, etc.  |
| Log warnings, not errors          | Errors are for truly fatal startup failures — tool failures are expected |

---

*Last updated: June 2026 — Mark 2 release*