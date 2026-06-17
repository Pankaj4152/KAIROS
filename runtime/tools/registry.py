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

# ── registry ───────────────────────────────────────────────────────────────────

REGISTRY: dict[str, dict] = {

    "web_search": {
        "domain": "search",
        "description": (
            "Search the web for current information. Use when the user asks about "
            "recent events, facts you may not know, or anything requiring up-to-date data."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                    "description": "The search query string",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "handler": _load_web_search,   # callable that returns the actual handler
        "enabled": True,
        "requires_env": [],            # duckduckgo default needs no key
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
            "Interact with the user's primary Google Calendar. "
            "Supports listing, free-text searching, creating, updating, and deleting events. "
            "When updating or deleting, try using a specific 'event_id' if available from past turns, "
            "otherwise use the 'query' text parameters to locate the subject event dynamically."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_events", "search_events", "create_event", "delete_event", "update_event"],
                    "description": "The target action to perform.",
                },
                "max_results": {
                    "type": "integer",
                    "default": 10,
                },
                "query": {
                    "type": "string",
                    "description": "Text query used to find matching events for search, deletion, or modifications.",
                },
                "event_id": {
                    "type": "string",
                    "description": "The exact alphanumeric target ID assigned by Google.",
                },
                "summary": {"type": "string", "description": "Title for a new event."},
                "start_time": {"type": "string", "description": "ISO start time string."},
                "end_time": {"type": "string", "description": "ISO end time string."},
                "description": {"type": "string"},
                "new_summary": {"type": "string", "description": "Used to overwrite titles in update_event."},
                "new_description": {"type": "string", "description": "Used to overwrite details in update_event."},
                "new_start_time": {"type": "string", "description": "Used to adjust timing in update_event."},
                "new_end_time": {"type": "string", "description": "Used to adjust timing in update_event."}
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        "handler": _load_google_calendar,
        "enabled": True,
        "requires_env": [],
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