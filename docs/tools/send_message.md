# send_message

**← [Back to Tool Guide](../TOOL_GUIDE.md)**

**File:** `runtime/tools/messaging.py`
**Env vars required:** `TELEGRAM_BOT_TOKEN`, `TELEGRAM_USER_ID`
**Execution pattern:** agentic

## When it's used

- The user explicitly asks Kairos to send them something on Telegram
- Kairos wants to push a proactive alert (reminder, briefing chunk, etc.)
- Classifier sets `tools_needed: [send_message]`

## Parameters

| Parameter | Type   | Required | Constraints     | Description                          |
|-----------|--------|----------|------------------|---------------------------------------|
| `message` | string | yes      | max 4096 chars   | The message text to send to the user |

> **4096-char hard limit** — Telegram's API rejects longer messages. If content is longer, the tool returns an error string instructing the LLM to chunk it and call again with `Part 1 of N` framing.

## Example LLM call

```json
{
  "name": "send_message",
  "input": {
    "message": "Reminder: your 3pm standup starts in 10 minutes."
  }
}
```

## Return format

```
SUCCESS: Message sent to Telegram. You must now reply to the user in text confirming it was sent, and DO NOT call this tool again.
```

The success string carries an explicit directive preventing the LLM from calling `send_message` again on the same turn — local tier models tend to loop on tool success without this.

## Failure modes

| Condition                     | Return value                                                                                                        |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------|
| Message exceeds 4096 chars     | `Error: Message too long. Telegram max is 4096 characters. Your message was N characters. Please shorten it or intelligently chunk it into multiple separate tool calls (e.g. 'Part 1 of 2').` |
| `TELEGRAM_BOT_TOKEN` or `TELEGRAM_USER_ID` not set | `Error: TELEGRAM_BOT_TOKEN or TELEGRAM_USER_ID is missing from environment variables.`                |
| Telegram API network error     | `Error sending message to Telegram API: <detail>`                                                                    |

## Notes

- Supports emoji, newlines, and standard Telegram text formatting
- Does not support inline keyboards, photos, or files — text only
- A fresh `Bot` instance is created per call (fire-and-forget) rather than reusing a long-lived connection
- This is currently the only tool in the registry that proactively reaches the user outside the normal response stream
