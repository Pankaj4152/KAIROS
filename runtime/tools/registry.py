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

def _load_weather():
    from tools.weather import weather
    return weather

def _load_finance():
    from tools.finance import finance
    return finance

def _load_tasks():
    from tools.tasks import tasks
    return tasks
 
 
def _load_spending():
    from tools.spending import spending
    return spending

def _load_habits():
    from tools.habits import habits
    return habits
 
 
def _load_notes():
    from tools.notes import notes
    return notes


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

    "finance": {
        "domain": "finance",
        "description": (
            "Retrieve stock, ETF, and cryptocurrency pricing and financial data. "
            "Uses Yahoo Finance (no API key required). "
            "Actions: "
            "'quote' — current price, change vs previous close, pre/post-market price, "
            "day range, volume, 52-week range, 50/200-day moving averages, "
            "market cap, P/E ratio, dividend yield (default); "
            "'history' — OHLCV candle history for a configurable period "
            "(1d / 5d / 1mo / 3mo / 6mo / 1y / 2y / 5y / max); "
            "'search' — find a ticker symbol by company name or keyword. "
            "Ticker format: US stocks use bare symbol (AAPL, TSLA, SPY); "
            "Indian NSE stocks append .NS (INFY.NS, RELIANCE.NS); "
            "Indian BSE stocks append .BO; "
            "Crypto uses bare symbol (BTC, ETH, SOL — auto-converted to BTC-USD etc.); "
            "Company names like 'Apple' or 'Reliance Industries' are auto-resolved to tickers."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 100,
                    "description": (
                        "Ticker symbol or company name. "
                        "Examples: 'AAPL', 'INFY.NS', 'BTC', 'ETH', 'Apple Inc', 'Reliance Industries'. "
                        "For Indian stocks use NSE suffix: TATAMOTORS.NS, HDFCBANK.NS."
                    ),
                },
                "action": {
                    "type": "string",
                    "enum": ["quote", "history", "search"],
                    "description": "Which data to retrieve. Default 'quote'.",
                },
                "period": {
                    "type": "string",
                    "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                    "description": (
                        "History period for action='history'. Default '5d'. "
                        "Interval is auto-selected per period for concise output."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "handler": _load_finance,
        "enabled": True,
        "requires_env": [],   # Yahoo Finance is keyless
    },
    
    "weather": {
        "domain": "weather",
        "description": (
            "Get weather conditions and forecasts for any location worldwide. "
            "Uses Open-Meteo API (free, no API key required). "
            "Actions: "
            "'current' — temperature, feels like, humidity, wind speed/direction/gusts, "
            "UV index, visibility, precipitation, pressure (default); "
            "'forecast' — daily min/max temp, condition, precipitation sum and chance, "
            "wind max, UV index, sunrise/sunset for 1–7 days; "
            "'hourly' — hour-by-hour temp, condition, rain chance, wind, UV for 1–48 hours. "
            "Supports metric (°C, km/h, mm) and imperial (°F, mph, inch) units."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                    "description": (
                        "City name, region, or country. Be specific for accuracy. "
                        "Examples: 'Mumbai', 'New Delhi', 'London UK', 'New York City', "
                        "'Bengaluru Karnataka', 'Tokyo Japan'."
                    ),
                },
                "action": {
                    "type": "string",
                    "enum": ["current", "forecast", "hourly"],
                    "description": "Which weather data to return. Default 'current'.",
                },
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 7,
                    "description": "Number of forecast days for action='forecast'. Default 3.",
                },
                "hours": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 48,
                    "description": "Number of hours for action='hourly'. Default 24.",
                },
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial"],
                    "description": (
                        "Unit system. 'metric' = °C, km/h, mm (default). "
                        "'imperial' = °F, mph, inch."
                    ),
                },
            },
            "required": ["location"],
            "additionalProperties": False,
        },
        "handler": _load_weather,
        "enabled": True,
        "requires_env": [],   # Open-Meteo is keyless
    },
    
    "tasks": {
        "domain": "tasks",
        "description": (
            "Create, list, update, complete, delete, and search the user's tasks, "
            "and view task statistics. Tasks are separate from Google Calendar events "
            "(Phase 1) — use this for to-dos and work items, use google_calendar for "
            "meetings and scheduled events. "
            "Actions: "
            "'create' — add a new task (title required; optional due_date, project, priority); "
            "'list' — list tasks, defaults to open tasks sorted by priority then due date, "
            "filterable by status/priority/due_before/due_after/project; "
            "'update' — change title, due_date, project, or priority on an existing task "
            "(requires task_id); "
            "'complete' — mark a task done (requires task_id; pass undo=true to reopen); "
            "'delete' — permanently remove a task (requires task_id); "
            "'search' — full-text search over title and project (requires query); "
            "'stats' — counts by status/priority, overdue count, due-today count."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "update", "complete", "delete", "search", "stats"],
                    "description": "Which task operation to perform.",
                },
                "task_id": {
                    "type": "integer",
                    "description": "Target task ID. Required for update, complete, delete.",
                },
                "title": {
                    "type": "string",
                    "description": "Task title. Required for create; optional for update.",
                },
                "due_date": {
                    "type": "string",
                    "description": (
                        "ISO date 'YYYY-MM-DD'. For update, pass an empty string to clear "
                        "an existing due date."
                    ),
                },
                "project": {
                    "type": "string",
                    "description": "Optional project or category tag, e.g. 'kairos', 'college'.",
                },
                "priority": {
                    "type": "integer",
                    "enum": [1, 2, 3],
                    "description": "1=low, 2=normal (default), 3=high.",
                },
                "status": {
                    "type": "string",
                    "enum": ["open", "done"],
                    "description": "Filter for list action. Defaults to open tasks only.",
                },
                "due_before": {
                    "type": "string",
                    "description": "Filter for list: tasks due on or before this ISO date.",
                },
                "due_after": {
                    "type": "string",
                    "description": "Filter for list: tasks due on or after this ISO date.",
                },
                "query": {
                    "type": "string",
                    "description": "Search text for action='search'.",
                },
                "undo": {
                    "type": "boolean",
                    "description": "For action='complete': pass true to reopen a completed task.",
                },
                "days": {
                    "type": "integer",
                    "description": "Lookback window in days for action='stats'. Default 7.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows returned for list/search. Default 20, max 100.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        "handler": _load_tasks,
        "enabled": True,
        "requires_env": [],   # uses the local kairos.db, no external creds
    },
    
    "spending": {
        "domain": "spending",
        "description": (
            "Log and track personal spending. Manual logging only — the user tells "
            "you what they spent and you record it. There is no automatic bank/UPI "
            "sync in this version. "
            "Actions: "
            "'log' — record a new expense (amount and category required; optional "
            "merchant, date, notes); "
            "'list' — list expenses newest first, filterable by category/month/date range; "
            "'update' — change amount, category, merchant, date, or notes on an existing "
            "expense (requires expense_id); "
            "'delete' — remove a mistakenly logged expense (requires expense_id); "
            "'budget' — check spend-to-date vs budget for a category (or overall) in a "
            "given month; pass set_amount to set or update the budget instead of checking it; "
            "'report' — category breakdown with totals and percentages for a period "
            "(this_month, last_month, last_7_days, last_30_days, or explicit 'YYYY-MM'). "
            "Valid categories: food, transport, utilities, entertainment, health, "
            "shopping, subscriptions, rent, education, other."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["log", "list", "update", "delete", "budget", "report"],
                    "description": "Which spending operation to perform.",
                },
                "expense_id": {
                    "type": "integer",
                    "description": "Target expense ID. Required for update, delete.",
                },
                "amount": {
                    "type": "number",
                    "description": "Expense amount (positive number). Required for log.",
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "food", "transport", "utilities", "entertainment",
                        "health", "shopping", "subscriptions", "rent", "education", "other",
                    ],
                    "description": "Expense category. Required for log.",
                },
                "merchant": {
                    "type": "string",
                    "description": "Optional merchant or vendor name, e.g. 'Swiggy', 'BigBasket'.",
                },
                "date": {
                    "type": "string",
                    "description": "ISO date 'YYYY-MM-DD'. Defaults to today for log.",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional free-text note, e.g. 'team lunch'.",
                },
                "month": {
                    "type": "string",
                    "description": "'YYYY-MM' filter for list, or target month for budget. Defaults to current month for budget.",
                },
                "date_from": {
                    "type": "string",
                    "description": "Filter for list: expenses on or after this ISO date.",
                },
                "date_to": {
                    "type": "string",
                    "description": "Filter for list: expenses on or before this ISO date.",
                },
                "set_amount": {
                    "type": "number",
                    "description": "For action='budget': set/update the budget to this amount instead of checking it.",
                },
                "period": {
                    "type": "string",
                    "description": (
                        "For action='report': 'this_month', 'last_month', 'last_7_days', "
                        "'last_30_days', or explicit 'YYYY-MM'. Default 'this_month'."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows returned for list. Default 20, max 200.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        "handler": _load_spending,
        "enabled": True,
        "requires_env": [],   # uses the local kairos.db, no external creds
    },

    "habits": {
        "domain": "habits",
        "description": (
            "Track habits: create them, check in daily completions, view streaks "
            "and consistency, and see a 30-day calendar heatmap. "
            "Actions: "
            "'create' — add a new habit (name required; optional target_frequency, "
            "one of daily/weekdays/3x_week/5x_week/weekly, defaults to daily); "
            "'list' — list all habits with current streak and last checkin date; "
            "'checkin' — log a completion for today, or pass date to backfill a past "
            "day (requires habit_id); "
            "'undo_checkin' — remove a checkin to fix a mistake (requires habit_id); "
            "'stats' — streak, 30-day consistency percentage, and an ASCII heatmap "
            "for one habit (requires habit_id); "
            "'delete' — permanently remove a habit and its full checkin history "
            "(requires habit_id)."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "checkin", "undo_checkin", "stats", "delete"],
                    "description": "Which habit operation to perform.",
                },
                "habit_id": {
                    "type": "integer",
                    "description": "Target habit ID. Required for checkin, undo_checkin, stats, delete.",
                },
                "name": {
                    "type": "string",
                    "description": "Habit name. Required for create, e.g. 'meditation', 'gym', 'DSA practice'.",
                },
                "target_frequency": {
                    "type": "string",
                    "enum": ["daily", "weekdays", "3x_week", "5x_week", "weekly"],
                    "description": "How often the habit is expected. Defaults to 'daily' for create.",
                },
                "date": {
                    "type": "string",
                    "description": (
                        "ISO date 'YYYY-MM-DD' for checkin/undo_checkin. Defaults to today. "
                        "Pass a past date to backfill a missed checkin."
                    ),
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        "handler": _load_habits,
        "enabled": True,
        "requires_env": [],   # uses the local kairos.db, no external creds
    },
    
    "notes": {
        "domain": "notes",
        "description": (
            "Capture, search, and manage notes — fleeting thoughts, decisions, "
            "and reference material the user explicitly wants to save. This is "
            "different from automatic conversation memory: notes are deliberately "
            "written and titled by the user, not extracted from chat turns. "
            "Actions: "
            "'create' — write a new note (title required; optional body, tags as "
            "comma-separated string, and link_type+link_id to attach it to a task "
            "or habit); "
            "'list' — list recent notes, newest-updated first, optionally filtered "
            "by a single tag; "
            "'get' — fetch one note's full content (requires note_id); "
            "'search' — fast keyword search over title and body (requires query); "
            "'semantic_search' — meaning-based search that finds related notes even "
            "without exact keyword overlap, using the same embedding pipeline as "
            "conversation memory (requires query); "
            "'update' — change title, body, or tags on an existing note (requires "
            "note_id); "
            "'delete' — permanently remove a note (requires note_id); "
            "'link' — attach an existing note to a task or habit (requires note_id, "
            "link_type, link_id)."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "create", "list", "get", "search", "semantic_search",
                        "update", "delete", "link",
                    ],
                    "description": "Which notes operation to perform.",
                },
                "note_id": {
                    "type": "integer",
                    "description": "Target note ID. Required for get, update, delete, link.",
                },
                "title": {
                    "type": "string",
                    "description": "Note title. Required for create; optional for update.",
                },
                "body": {
                    "type": "string",
                    "description": "Note content. Optional for create (defaults empty) and update.",
                },
                "tags": {
                    "type": "string",
                    "description": "Comma-separated tags, e.g. 'ml,research,kairos'. For create/update.",
                },
                "tag": {
                    "type": "string",
                    "description": "Single tag to filter by for action='list'.",
                },
                "query": {
                    "type": "string",
                    "description": "Search text for action='search' or 'semantic_search'.",
                },
                "link_type": {
                    "type": "string",
                    "enum": ["task", "habit"],
                    "description": "Entity type to link the note to. For create or link.",
                },
                "link_id": {
                    "type": "integer",
                    "description": "ID of the task or habit to link the note to. For create or link.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows for list/search. Default 20, max 100.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max results for semantic_search. Default 5.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        "handler": _load_notes,
        "enabled": True,
        "requires_env": [],   # keyword search needs nothing; semantic_search reuses
                            # vector_store.py's existing LITELLM_BASE_URL config
    },
    


    # "weather": {
    #     "domain": "weather",
    #     "description": (
    #         "Retrieve current weather conditions and a 3-day forecast for any location worldwide. "
    #         "Input: a location/city name, and optionally whether a multi-day forecast is needed. "
    #         "Use this tool whenever the user asks about the weather, temperature, rain, or general forecast in a specific city/region."
    #     ),
    #     "schema": {
    #         "type": "object",
    #         "properties": {
    #             "location": {
    #                 "type": "string",
    #                 "minLength": 1,
    #                 "description": "The city or location name (e.g. 'London', 'Paris', 'Tokyo').",
    #             },
    #             "forecast": {
    #                 "type": "boolean",
    #                 "default": False,
    #                 "description": "If true, includes daily min/max temperature, precipitation, and conditions for the next 3 days.",
    #             },
    #         },
    #         "required": ["location"],
    #         "additionalProperties": False,
    #     },
    #     "handler":      _load_weather,
    #     "enabled":      True,
    #     "requires_env": [],
    # },

    # "finance": {
    #     "domain": "finance",
    #     "description": (
    #         "Retrieve current pricing, currency, exchange, change stats, and optional 5-day daily open/close chart history "
    #         "for stocks (e.g. AAPL, MSFT, TSLA) or cryptocurrencies (e.g. BTC, ETH, SOL). "
    #         "Input: a stock ticker/symbol, crypto coin symbol, or company name (e.g. 'Apple', 'Bitcoin', 'Tesla'). "
    #         "Use this tool whenever the user asks for stock quotes, crypto price check, market stats, or daily chart trends."
    #     ),
    #     "schema": {
    #         "type": "object",
    #         "properties": {
    #             "query": {
    #                 "type": "string",
    #                 "minLength": 1,
    #                 "description": "Stock symbol, crypto ticker, or company/asset name (e.g., 'AAPL', 'BTC', 'Microsoft').",
    #             },
    #             "history": {
    #                 "type": "boolean",
    #                 "default": False,
    #                 "description": "If true, returns daily open and close prices for the last 5 trading days.",
    #             },
    #         },
    #         "required": ["query"],
    #         "additionalProperties": False,
    #     },
    #     "handler":      _load_finance,
    #     "enabled":      True,
    #     "requires_env": [],
    # },

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