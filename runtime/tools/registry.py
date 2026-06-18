"""
Tool registry — single source of truth for every tool Kairos can call.

Each entry declares:
    description     what the tool does (shown to LLM)
    schema          JSON schema for input validation (enforced before dispatch)
    handler         async callable — must accept **kwargs matching schema props
    enabled         False = never dispatched, even if LLM requests it
    requires_env    list of env vars that must be set for this tool to be eligible
                    checked at startup via check_eligibility(), not at call time

Adding a new tool:
    1. Write the handler in tools/<name>.py
    2. Add an entry here
    3. That's it — executor picks it up automatically
"""

import logging
import os

logger = logging.getLogger(__name__)


# ── lazy imports ───────────────────────────────────────────────────────────────
# Import inside functions so a broken tool file doesn't kill the whole registry

def _load_web_search():
    from tools.web_search import web_search
    return web_search

def _load_send_message():
    from tools.messaging import send_message
    return send_message

def _load_google_calendar():
    from tools.google_calendar import google_calendar_action
    return google_calendar_action

def _load_google_keep():
    from tools.google_keep import google_keep_action
    return google_keep_action

def _load_check_gmail():
    from tools.gmail_check import check_gmail
    return check_gmail

def _load_gmail_actions():
    from tools.gmail_actions import gmail_actions
    return gmail_actions

# ── registry ───────────────────────────────────────────────────────────────────

REGISTRY: dict[str, dict] = {

    "web_search": {
        "domain": "search",
        "description": (
            "Search the web for current information. "
            "Use when the user asks about recent events, news, facts that may have changed, "
            "or anything requiring up-to-date data. "
            "Input: a concise search query string (max 200 chars). "
            "Output: a list of sourced snippets — cite them, do not paraphrase freely."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                    "description": "Concise search query. Be specific — avoid vague queries like 'tell me about X'.",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                    "description": "Number of results to return (default 5).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "handler":      _load_web_search,   # factory — called once per process
        "enabled":      True,
        "requires_env": [],                 # duckduckgo default needs no key
        "timeout_sec":  15.0,
    },

    "send_message": {
        "domain": "messaging",
        "description": (
            "Push a direct message to the user's Telegram. Use this whenever the user asks you to send them something "
            "on Telegram, OR to proactively alert them. Supports emojis and standard formatting. "
            "CRITICAL: Telegram has a hard limit of 4096 characters. If your content is longer, "
            "chunk it into smaller parts (e.g., 'Part 1...') and call this tool multiple times."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "maxLength": 4096,
                    "description": "The message text to send to the user.",
                }
            },
            "required": ["message"],
            "additionalProperties": False,
        },
        "handler": _load_send_message,
        "enabled": True,
        "requires_env": ["TELEGRAM_BOT_TOKEN", "TELEGRAM_USER_ID"],
    },
    
    "google_calendar": {
        "domain": "calendar",
        "description": (
            "Interact with the user's Google Calendar. "
            "Actions: "
            "'list_events' — upcoming events with times, locations, and IDs; "
            "'get_event' — full details of one event including notes, attendees, and Meet link (requires 'event_id'); "
            "'search_events' — free-text search across all events past and future (requires 'query'); "
            "'create_event' — create a new event, returns the new event ID (requires summary, start_time, end_time); "
            "'update_event' — patch specific fields without touching others (requires event_id or query, plus new_* fields); "
            "'delete_event' — delete by event_id, or search by query (shows matches if ambiguous); "
            "'list_calendars' — list all calendars and their IDs. "
            "Event IDs appear in list_events and search_events output as 'ID: <value>' — use these for get/update/delete. "
            "Datetimes use ISO 8601 format: '2026-06-20T15:00:00' (no offset needed — timezone from user settings)."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "list_events", "get_event", "search_events",
                        "create_event", "update_event", "delete_event",
                        "list_calendars",
                    ],
                    "description": "Which calendar operation to perform.",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                    "description": "Max events to return for list_events and search_events.",
                },
                "query": {
                    "type": "string",
                    "maxLength": 300,
                    "description": "Free-text search string. Required for search_events. Optional for update_event and delete_event when event_id is unknown.",
                },
                "event_id": {
                    "type": "string",
                    "description": "Google Calendar event ID from a prior list_events or search_events result. Required for get_event. Preferred over 'query' for update_event and delete_event.",
                },
                "calendar_id": {
                    "type": "string",
                    "description": "Calendar to operate on. Defaults to the primary calendar. Get IDs from list_calendars.",
                },
                "summary": {
                    "type": "string",
                    "description": "Event title. Required for create_event.",
                },
                "description": {
                    "type": "string",
                    "description": "Event body / notes. Optional for create_event.",
                },
                "location": {
                    "type": "string",
                    "description": "Location string. Optional for create_event.",
                },
                "start_time": {
                    "type": "string",
                    "description": "ISO 8601 start datetime. Required for create_event. Example: '2026-06-20T15:00:00'.",
                },
                "end_time": {
                    "type": "string",
                    "description": "ISO 8601 end datetime. Required for create_event.",
                },
                "attendee_emails": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Email addresses to invite. Optional for create_event. Invite emails are sent automatically.",
                },
                "new_summary": {
                    "type": "string",
                    "description": "Replacement title for update_event.",
                },
                "new_description": {
                    "type": "string",
                    "description": "Replacement description for update_event.",
                },
                "new_start_time": {
                    "type": "string",
                    "description": "Replacement start datetime for update_event.",
                },
                "new_end_time": {
                    "type": "string",
                    "description": "Replacement end datetime for update_event.",
                },
                "new_location": {
                    "type": "string",
                    "description": "Replacement location for update_event.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        "handler": _load_google_calendar,
        "enabled": True,
        "requires_env": [],   # auth via token.json / credentials.json, not env vars
    },
    
    "check_gmail": {
        "description": (
            "Read and manage the user's Gmail inbox via IMAP. "
            "Actions: "
            "'list_unread' — headers of recent unread emails (default); "
            "'list_recent' — headers of recent emails regardless of read status; "
            "'search' — search inbox by keyword (provide 'query'); "
            "'get_body' — full readable body of one email (provide 'uid' from a previous list/search); "
            "'mark_read' — mark one email as read (provide 'uid'); "
            "'count_unread' — unread count only, fastest option. "
            "UIDs appear in list/search output as 'UID: <value>' — copy the value to pass as 'uid'."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_unread", "list_recent", "search", "get_body", "mark_read", "count_unread"],
                    "description": "The Gmail operation to perform.",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                    "description": "Max emails to return for list/search actions.",
                },
                "query": {
                    "type": "string",
                    "maxLength": 200,
                    "description": "Keyword to search for. Required when action='search'.",
                },
                "uid": {
                    "type": "string",
                    "description": (
                        "Email UID from a previous list_unread, list_recent, or search result. "
                        "Required when action='get_body' or action='mark_read'."
                    ),
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        "handler": _load_check_gmail,
        "enabled": True,
        "requires_env": ["GMAIL_USER", "GMAIL_APP_PASSWORD"],
    },
    
    "gmail_actions": {
        "domain": "email",
        "description": (
            "Send, reply to, forward, delete, archive, move, and manage emails in Gmail. "
            "This is the WRITE tool — use gmail_check for reading/searching. "
            "Actions: "
            "'send' — compose and send a new email (requires to, subject, body); "
            "'reply' — reply to an email by UID from gmail_check output (requires uid, body); "
            "'reply_all' — reply to all recipients of an email (requires uid, body); "
            "'forward' — forward an email to new recipients (requires uid, to); "
            "'delete' — move email to Trash by UID (recoverable, requires uid); "
            "'delete_permanent' — permanently delete email, NO TRASH, UNRECOVERABLE — always confirm with user first (requires uid); "
            "'archive' — move email out of Inbox to All Mail without deleting (requires uid); "
            "'move' — move email to a specific folder or label (requires uid, destination); "
            "'mark_unread' — mark a read email as unread (requires uid); "
            "'list_folders' — list all folders and custom labels on this account; "
            "'create_draft' — save a draft to Gmail Drafts folder (requires to, subject, body). "
            "UIDs come from gmail_check tool output — shown as 'UID: <value>'. "
            "Addresses accept 'Name <email>' or plain 'email@domain.com' format. "
            "Multiple addresses are comma-separated."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "send", "reply", "reply_all", "forward",
                        "delete", "delete_permanent", "archive", "move",
                        "mark_unread", "list_folders", "create_draft",
                    ],
                    "description": "Which Gmail operation to perform.",
                },
                "to": {
                    "type": "string",
                    "description": (
                        "Recipient address(es), comma-separated. "
                        "Accepts 'Name <email>' or plain 'email@domain.com'. "
                        "Required for: send, forward, create_draft."
                    ),
                },
                "subject": {
                    "type": "string",
                    "maxLength": 998,
                    "description": "Email subject line. Required for: send, create_draft.",
                },
                "body": {
                    "type": "string",
                    "description": (
                        "Plain text email body. "
                        "Required for: send, reply, reply_all, create_draft. "
                        "Optional for forward (used as a note before the quoted original)."
                    ),
                },
                "cc": {
                    "type": "string",
                    "description": "CC address(es), comma-separated. Optional for send, create_draft.",
                },
                "bcc": {
                    "type": "string",
                    "description": "BCC address(es), comma-separated. Optional for send only. Not visible to recipients.",
                },
                "uid": {
                    "type": "string",
                    "description": (
                        "IMAP UID of the email to act on. "
                        "Get UIDs from gmail_check output — shown as 'UID: <value>'. "
                        "Required for: reply, reply_all, forward, delete, delete_permanent, "
                        "archive, move, mark_unread."
                    ),
                },
                "reply_all": {
                    "type": "boolean",
                    "description": (
                        "If true, reply to all original recipients (From + To + CC). "
                        "Only used for reply action. Default false."
                    ),
                },
                "extra_to": {
                    "type": "string",
                    "description": "Additional reply recipients, comma-separated. Optional for reply/reply_all.",
                },
                "note": {
                    "type": "string",
                    "description": "Text to prepend before the quoted original in a forward. Optional.",
                },
                "permanent": {
                    "type": "boolean",
                    "description": (
                        "For delete_permanent only. Must be true to confirm permanent deletion. "
                        "ALWAYS confirm with the user before setting this to true."
                    ),
                },
                "destination": {
                    "type": "string",
                    "description": (
                        "Target folder for move action. "
                        "Use short names: inbox, sent, drafts, spam, trash, archive, starred. "
                        "Or full IMAP path like '[Gmail]/All Mail' or a custom label name. "
                        "Use list_folders to see all available options. "
                        "Required for: move."
                    ),
                },
                "folder": {
                    "type": "string",
                    "description": (
                        "Source mailbox the UID lives in. Default 'inbox'. "
                        "Set this when the email is NOT in INBOX — e.g. 'sent' for sent mail, "
                        "'spam' for spam folder. Use short names or full IMAP path."
                    ),
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        "handler": _load_gmail_actions,
        "enabled": True,
        "requires_env": ["GMAIL_USER", "GMAIL_APP_PASSWORD"],
    },

    # "google_keep": {
    #     "domain": "notes",
    #     "description": (
    #         "Interact with the user's personal Google Keep scratchpad account. "
    #         "Use this tool for rapid text capture, keeping reminders, logging transient "
    #         "brain dumps, creating lists, or appending thoughts to an existing note text block."
    #     ),
    #     "schema": {
    #         "type": "object",
    #         "properties": {
    #             "action": {
    #                 "type": "string",
    #                 "enum": ["list_notes", "create_note", "append_to_note"],
    #                 "description": "The scratchpad operation to execute.",
    #             },
    #             "max_results": {
    #                 "type": "integer",
    #                 "default": 5,
    #             },
    #             "title": {
    #                 "type": "string",
    #                 "description": "The title of the note (used for targeting or creation).",
    #             },
    #             "text": {
    #                 "type": "string",
    #                 "description": "The text string content to write or append inside the target note.",
    #             }
    #         },
    #         "required": ["action"],
    #         "additionalProperties": False,
    #     },
    #     "handler": _load_google_keep,
    #     "enabled": True,
    #     "requires_env": ["GOOGLE_USERNAME", "GOOGLE_APP_PASSWORD"],
    # },
}


# ── eligibility check ──────────────────────────────────────────────────────────

def check_eligibility() -> dict[str, bool]:
    """
    Called once at startup. Logs a warning for every tool that is registered
    but cannot run due to missing env vars. Returns eligibility map.

    Executor uses this map — ineligible tools are rejected before handler load.
    """
    eligibility: dict[str, bool] = {}

    for name, tool in REGISTRY.items():
        if not tool["enabled"]:
            eligibility[name] = False
            logger.debug("Tool '%s' is disabled", name)
            continue

        missing = [var for var in tool["requires_env"] if not os.getenv(var)]
        if missing:
            eligibility[name] = False
            logger.warning(
                "Tool '%s' ineligible — missing env vars: %s", name, missing
            )
        else:
            eligibility[name] = True
            logger.info("Tool '%s' ready", name)

    return eligibility


# Module-level eligibility map — populated on first import
_eligibility: dict[str, bool] | None = None

def get_eligibility() -> dict[str, bool]:
    global _eligibility
    if _eligibility is None:
        _eligibility = check_eligibility()
    return _eligibility


def get_tool_schemas() -> list[dict]:
    """
    Returns tool descriptions in the shape LiteLLM / Anthropic expects.
    Only includes enabled + eligible tools.
    Used by context assembler to tell the LLM what tools exist.
    """
    eligibility = get_eligibility()
    schemas = []
    for name, tool in REGISTRY.items():
        if not eligibility.get(name):
            continue
        schemas.append({
            "name": name,
            "description": tool["description"],
            "input_schema": tool["schema"],
        })
    return schemas