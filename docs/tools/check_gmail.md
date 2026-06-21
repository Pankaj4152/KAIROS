# check_gmail

**← [Back to Tool Guide](../TOOL_GUIDE.md)**

**File:** `runtime/tools/gmail_check.py`
**Env vars required:** `GMAIL_USER`, `GMAIL_APP_PASSWORD`
**Execution pattern:** agentic
**See also:** [gmail_actions](gmail_actions.md) — the write-side companion (send/reply/delete/move). This tool is **read-only**.

> `GMAIL_APP_PASSWORD` is a 16-character Google App Password — not your account password. Generate one at myaccount.google.com/apppasswords. IMAP must be enabled in Gmail settings (Settings → See all settings → Forwarding and POP/IMAP → Enable IMAP).

## When it's used

- User asks to check email, look for alerts, or summarize their inbox
- User wants to search by keyword or read a specific email
- Classifier sets `tools_needed: [check_gmail]`

## Parameters

| Parameter     | Type    | Required for                            | Constraints      | Description |
|---------------|---------|------------------------------------------|-------------------|--------------|
| `action`      | string  | all                                       | enum              | Which operation to perform |
| `max_results` | integer | `list_unread`, `list_recent`, `search`    | 1–20, default 5   | Max emails to return (clamped, not rejected) |
| `query`       | string  | `search`                                  | max 200 chars     | Keyword — matches subject, body, and headers via IMAP `TEXT` search |
| `uid`         | string  | `get_body`, `mark_read`                   | from prior output | Email UID from a previous list/search result |

## Actions

| Action         | What it does                                                  | Required params | Network cost |
|-----------------|------------------------------------------------------------------|-------------------|----------------|
| `list_unread`   | Headers of unread emails, most recent first (default)            | —                 | Low |
| `list_recent`   | Headers of recent emails, read or unread, most recent first      | —                 | Low |
| `search`        | Keyword search — subject, body, headers                          | `query`           | Low |
| `get_body`      | Full readable text body of one email (HTML stripped to plain text, truncated at 1500 chars) | `uid` | Medium |
| `mark_read`     | Mark one email as read (sets the `\Seen` IMAP flag)               | `uid`             | Low |
| `count_unread`  | Unread count only — fastest, no header fetch                      | —                 | Minimal |

## Example LLM calls

**Check unread (most common):**
```json
{ "name": "check_gmail", "input": { "action": "list_unread", "max_results": 5 } }
```

**Search for a specific email:**
```json
{ "name": "check_gmail", "input": { "action": "search", "query": "GitHub deployment failed", "max_results": 3 } }
```

**Read full body (UID from prior output):**
```json
{ "name": "check_gmail", "input": { "action": "get_body", "uid": "18943" } }
```

**Just the count:**
```json
{ "name": "check_gmail", "input": { "action": "count_unread" } }
```

## Return format (list_unread)

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
```

The `UID:` value is what you pass to `get_body` or `mark_read`. UIDs are stable across mailbox reindexing — they don't shift when other emails arrive or get deleted.

## Return format (get_body)

```
FROM: GitHub <noreply@github.com>
SUBJECT: [kairos] Deployment failed
DATE: Thu, 18 Jun 2026 09:14:22 +0530

BODY:
The workflow run "Deploy to prod" triggered by push to main failed.
...
[… truncated at 1500 chars]
```

## Failure modes

| Condition                                  | Return value |
|----------------------------------------------|-----------------|
| `GMAIL_USER`/`GMAIL_APP_PASSWORD` not set, or IMAP login fails | `Error: Gmail credentials missing or IMAP login failed.` |
| UID doesn't exist                             | `Could not fetch email with UID <n>. It may no longer exist.` |
| `search` with empty `query`                   | `Error: 'search' action requires a non-empty 'query' parameter.` |
| `get_body`/`mark_read` without `uid`          | `Error: '<action>' action requires a 'uid' parameter. Get UIDs from list_unread or search.` |
| Network/IMAP error mid-operation              | `Failed to retrieve emails: <detail>` |
| Unknown action                                | `Error: unknown action '<action>'. Valid actions: list_unread, list_recent, search, get_body, mark_read, count_unread.` |

## Typical multi-turn pattern

```
User: "Any important emails?"
  → check_gmail(action="list_unread", max_results=5)
  ← 5 headers with UIDs

User: "Read the GitHub one"
  → check_gmail(action="get_body", uid="18943")
  ← full body text

User: "Mark it read"
  → check_gmail(action="mark_read", uid="18943")
  ← "Marked email UID 18943 as read."
```

For replying, forwarding, deleting, or sending new mail, the LLM switches to [gmail_actions](gmail_actions.md) using the same UID.
