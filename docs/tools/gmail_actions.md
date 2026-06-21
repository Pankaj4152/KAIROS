# gmail_actions

**← [Back to Tool Guide](../TOOL_GUIDE.md)**

**File:** `runtime/tools/gmail_actions.py`
**Env vars required:** `GMAIL_USER`, `GMAIL_APP_PASSWORD` (same App Password used by [check_gmail](check_gmail.md))
**Execution pattern:** agentic
**Added:** Mark 3
**See also:** [check_gmail](check_gmail.md) — the read-only companion. UIDs from `check_gmail` output are used directly here.

This is the **write** side of Gmail integration: send, reply, forward, delete, archive, move, and manage emails. Reading and searching live in `check_gmail` — this split lets the registry (and you) gate write access independently of read access.

Auth uses two protocols with the same App Password:
- **SMTP** (`send`, `reply`, `reply_all`, `forward`, `create_draft`) — `smtp.gmail.com:587` with STARTTLS
- **IMAP** (`delete`, `delete_permanent`, `archive`, `move`, `mark_unread`, `list_folders`) — `imap.gmail.com:993` SSL

## When it's used

- User asks to send, reply to, forward, delete, or organize email
- Classifier sets `domains: [email]` and `tools_needed: [gmail_actions]`
- Almost always follows a `check_gmail` call that surfaced a UID

## Parameters

| Parameter      | Type    | Required for                                            | Description |
|-----------------|---------|------------------------------------------------------------|--------------|
| `action`        | string  | all                                                          | Which operation to perform |
| `to`            | string  | `send`, `forward`, `create_draft`                            | Comma-separated. Accepts `"Name <email>"` or plain `"email@domain.com"` |
| `subject`       | string  | `send`, `create_draft`                                        | Max 998 chars |
| `body`          | string  | `send`, `reply`, `reply_all`, `create_draft`; optional for `forward` (prepended note) | Plain text body |
| `cc`            | string  | optional, `send`/`create_draft`                              | Comma-separated |
| `bcc`           | string  | optional, `send` only                                        | Comma-separated; never appears in headers, only the SMTP envelope |
| `uid`           | string  | `reply`, `reply_all`, `forward`, `delete`, `delete_permanent`, `archive`, `move`, `mark_unread` | From `check_gmail` output |
| `reply_all`     | boolean | optional, `reply`                                             | If true, includes all original recipients (From + To + CC, self excluded). Default `false` |
| `extra_to`      | string  | optional, `reply`/`reply_all`                                  | Additional recipients beyond the original thread, comma-separated |
| `note`          | string  | optional, `forward`                                            | Text prepended before the quoted original |
| `permanent`     | boolean | `delete_permanent` confirmation                                | Must be `true`. **Always confirm with the user first** |
| `destination`   | string  | `move`                                                          | Short name (`inbox`/`sent`/`drafts`/`spam`/`trash`/`archive`/`starred`) or full IMAP path / custom label |
| `folder`        | string  | optional, all UID-based IMAP actions                            | Source mailbox the UID lives in. Default `"inbox"` — set explicitly when the email isn't in INBOX (e.g. `"sent"`, `"spam"`) |

## Actions

| Action              | What it does | Required params |
|----------------------|---------------|--------------------|
| `send`               | Compose and send a new email | `to`, `subject`, `body` |
| `reply`               | Reply to sender only, with proper `In-Reply-To`/`References` threading headers | `uid`, `body` |
| `reply_all`           | Reply to sender + all original To/CC recipients (self excluded) | `uid`, `body` |
| `forward`              | Forward an email, quoting the original body below an optional note | `uid`, `to` |
| `delete`                | Move to Trash — recoverable. Tries the IMAP `MOVE` extension first, falls back to COPY+`\Deleted`+expunge | `uid` |
| `delete_permanent`       | Set `\Deleted` and expunge immediately — **no Trash, unrecoverable** | `uid`, confirm with user before setting `permanent=true` |
| `archive`                 | Move to `[Gmail]/All Mail`, removing from Inbox without deleting | `uid` |
| `move`                      | Move to any named folder/label | `uid`, `destination` |
| `mark_unread`                 | Remove the `\Seen` flag | `uid` |
| `list_folders`                  | List standard Gmail folders + any custom labels on the account | — |
| `create_draft`                    | Save to `[Gmail]/Drafts` via IMAP APPEND with the `\Draft` flag | `to`, `subject`, `body` |

## Example LLM calls

**Send:**
```json
{ "name": "gmail_actions", "input": { "action": "send", "to": "alice@example.com", "subject": "Hello", "body": "Test body" } }
```

**Reply (uid from check_gmail):**
```json
{ "name": "gmail_actions", "input": { "action": "reply", "uid": "18943", "body": "Thanks, I'll take a look." } }
```

**Forward with a note:**
```json
{ "name": "gmail_actions", "input": { "action": "forward", "uid": "18943", "to": "teammate@example.com", "note": "FYI — can you check this?" } }
```

**Move to Trash:**
```json
{ "name": "gmail_actions", "input": { "action": "delete", "uid": "18943" } }
```

**Move to a custom label:**
```json
{ "name": "gmail_actions", "input": { "action": "move", "uid": "18943", "destination": "MyLabel" } }
```

**Permanent delete (only after explicit user confirmation):**
```json
{ "name": "gmail_actions", "input": { "action": "delete_permanent", "uid": "18943", "permanent": true } }
```

## Return format examples

```
Sent: "Hello" → alice@example.com
```
```
Reply sent: "Re: Original subject" → sender@example.com
```
```
Moved email UID 18943 to Trash.
```
```
Permanently deleted email UID 18943 from INBOX. This is unrecoverable and cannot be undone.
```

## Failure modes

| Condition                                     | Return value |
|--------------------------------------------------|-----------------|
| Unknown action                                    | `Error: unknown action '<action>'. Valid actions: send, reply, reply_all, forward, delete, delete_permanent, archive, move, mark_unread, list_folders, create_draft.` |
| Required param missing (validated before any network call) | e.g. `Error: 'reply' requires a 'uid'. Get UIDs from gmail_check list_unread/search output.` / `Error: 'send' requires a 'to' address.` / `Error: 'move' requires a 'destination' folder name.` |
| Invalid email address in `to`/`cc`/`bcc`           | `Error: No valid 'to' addresses found. Invalid: <input>.` or `Error: Invalid email addresses found: <list>. Fix them and retry.` |
| `GMAIL_USER`/`GMAIL_APP_PASSWORD` not set           | `Error: GMAIL_USER or GMAIL_APP_PASSWORD not set in environment.` |
| SMTP login/connection failure                        | `Error: Gmail SMTP login failed. Check GMAIL_USER and GMAIL_APP_PASSWORD.` |
| IMAP login/connection failure                         | `Error: Gmail IMAP login failed.` |
| Original message not found (reply/reply_all/forward)    | `Error: Could not fetch email with UID <uid>. It may not exist in INBOX.` |
| Reply recipients all resolve to self                       | `Error: Could not determine reply recipients from original email.` |
| `move` with unknown destination and IMAP rejects it           | `Error: Could not move email UID <uid> to '<dst>'. Check folder name.` |
| Unexpected exception                                             | `Unexpected error in gmail_actions '<action>': <detail>` |

## Notes

- `delete_permanent` is logged as a warning server-side (`logger.warning(...)`) even before execution — treat it as a destructive action and always get explicit user confirmation first, the same way you would for any irreversible operation.
- `destination`/`folder` short names map through `GMAIL_FOLDERS`: `inbox`→`INBOX`, `sent`→`[Gmail]/Sent Mail`, `drafts`→`[Gmail]/Drafts`, `spam`→`[Gmail]/Spam`, `trash`→`[Gmail]/Trash`, `archive`→`[Gmail]/All Mail`, `starred`→`[Gmail]/Starred`, `important`→`[Gmail]/Important`. Anything else is passed through as-is (custom labels).
- Threading (`In-Reply-To`/`References`) is handled automatically for `reply`/`reply_all` — you only need to supply `body`.
