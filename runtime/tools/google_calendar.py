"""
Google Calendar Tool — full CRUD for the user's Google Calendar.

Actions:
    list_events      — upcoming events, with end time / location / notes
    get_event        — full details of one event by ID (notes, attendees, location)
    search_events    — free-text search across all events (past + future)
    create_event     — create a new event; returns the new event ID
    update_event     — patch specific fields without clobbering the rest
    delete_event     — delete by event_id, or search + confirm by title
    list_calendars   — list all calendars the user has access to

Design rules (mirroring gmail_check.py):
    - Service object is created per call, closed in a finally block — no leaks.
    - Timezone is read from TIMEZONE env var (default Asia/Kolkata).
      All datetimes returned to the LLM are in the user's local timezone.
    - update_event uses events().patch() — only the fields you pass are changed.
      events().update() (full replace) is never used.
    - delete_event with a query searches for matching events and returns a
      confirmation list before deleting the first match — the LLM can show
      the user what it found before committing.
    - create_event returns the new event ID so a follow-up update/delete can
      reference it without another search round-trip.
    - Every action returns a plain string. Never raises to the LLM.

Auth:
    First run: place credentials.json (from Google Cloud Console) in the
    project root. The OAuth flow writes token.json there.
    Subsequent runs: token.json is refreshed automatically.

Env vars:
    TIMEZONE                  user's timezone (default: Asia/Kolkata)
    GOOGLE_CALENDAR_ID        calendar to operate on (default: primary)
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone as dt_timezone

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

try:
    from zoneinfo import ZoneInfo                    # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo          # pip install backports.zoneinfo

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar"]

# ── config ────────────────────────────────────────────────────────────────────

def _get_timezone() -> ZoneInfo:
    tz_name = os.getenv("TIMEZONE", "Asia/Kolkata")
    try:
        return ZoneInfo(tz_name)
    except Exception:
        logger.warning("Invalid TIMEZONE=%r, falling back to Asia/Kolkata", tz_name)
        return ZoneInfo("Asia/Kolkata")

def _get_calendar_id() -> str:
    return os.getenv("GOOGLE_CALENDAR_ID", "primary")

# ── auth ──────────────────────────────────────────────────────────────────────

def _get_service():
    """
    Build and return an authenticated Google Calendar service object.
    Refreshes the token automatically if expired.
    Raises FileNotFoundError if credentials.json is missing (setup error, not runtime).
    """
    creds = None
    token_path = "token.json"
    creds_path = os.getenv("GOOGLE_CALENDAR_CREDENTIALS_PATH", "credentials.json")

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(creds_path):
                raise FileNotFoundError(
                    f"Missing credentials file at '{creds_path}'. "
                    "Download it from Google Cloud Console and place it in the project root."
                )
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as f:
            f.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)


# ── datetime helpers ──────────────────────────────────────────────────────────

def _now_utc_iso() -> str:
    """Current time in UTC as an RFC 3339 string for API timeMin/timeMax."""
    return datetime.now(dt_timezone.utc).isoformat()


def _localize_dt_string(dt_str: str | None, tz: ZoneInfo) -> str:
    """
    Convert an ISO 8601 datetime string (from the API) to the user's local timezone.
    Handles both datetime strings (2026-06-20T15:00:00+05:30) and date-only strings
    (2026-06-20) which all-day events use.
    Returns a human-readable string like "Fri 20 Jun 2026, 3:00 PM".
    """
    if not dt_str:
        return "Unknown"

    # All-day event — date only, no time component
    if "T" not in dt_str:
        try:
            d = datetime.strptime(dt_str, "%Y-%m-%d")
            return d.strftime("%a %d %b %Y (all day)")
        except ValueError:
            return dt_str

    try:
        # Parse as aware datetime (Google always returns timezone-aware strings)
        dt = datetime.fromisoformat(dt_str)
        dt_local = dt.astimezone(tz)
        return dt_local.strftime("%a %d %b %Y, %I:%M %p")
    except (ValueError, OverflowError):
        return dt_str


def _format_event(event: dict, tz: ZoneInfo, show_full: bool = False) -> str:
    """
    Format one Google Calendar event dict into a readable string for the LLM.

    show_full=False → compact one-liner with ID, time, title, location
    show_full=True  → expanded with description, attendees, conference link
    """
    event_id    = event.get("id", "unknown")
    title       = event.get("summary", "(No title)")
    start       = _localize_dt_string(
        event["start"].get("dateTime") or event["start"].get("date"), tz
    )
    end         = _localize_dt_string(
        event["end"].get("dateTime") or event["end"].get("date"), tz
    )
    location    = event.get("location", "")
    description = event.get("description", "")
    status      = event.get("status", "confirmed")
    html_link   = event.get("htmlLink", "")

    if not show_full:
        parts = [f"ID: {event_id} | {start} → {end} | {title}"]
        if location:
            parts.append(f"  Location: {location}")
        return "\n".join(parts)

    # Full detail view
    lines = [
        f"Title:    {title}",
        f"ID:       {event_id}",
        f"Start:    {start}",
        f"End:      {end}",
        f"Status:   {status}",
    ]
    if location:
        lines.append(f"Location: {location}")
    if description:
        # Trim very long descriptions
        desc = description[:500] + "…" if len(description) > 500 else description
        lines.append(f"Notes:    {desc}")

    attendees = event.get("attendees", [])
    if attendees:
        names = ", ".join(
            a.get("displayName") or a.get("email", "unknown")
            for a in attendees[:8]
        )
        lines.append(f"Attendees ({len(attendees)}): {names}")

    conf = event.get("conferenceData", {})
    entry_points = conf.get("entryPoints", [])
    for ep in entry_points:
        if ep.get("entryPointType") == "video":
            lines.append(f"Meet link: {ep.get('uri', '')}")
            break

    if html_link:
        lines.append(f"Link:     {html_link}")

    return "\n".join(lines)


# ── actions (all sync — run inside asyncio.to_thread) ─────────────────────────

def _list_events(max_results: int, calendar_id: str, tz: ZoneInfo) -> str:
    """Upcoming events from now, soonest first."""
    service = _get_service()
    try:
        result = service.events().list(
            calendarId=calendar_id,
            timeMin=_now_utc_iso(),
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        events = result.get("items", [])
        if not events:
            return "No upcoming events found."

        lines = [f"Upcoming events ({len(events)} shown):"]
        for e in events:
            lines.append(_format_event(e, tz))
        return "\n\n".join(lines)
    except HttpError as e:
        return f"Error listing events: {e}"
    finally:
        service.close()


def _get_event(event_id: str, calendar_id: str, tz: ZoneInfo) -> str:
    """Full details of a single event by ID."""
    service = _get_service()
    try:
        event = service.events().get(
            calendarId=calendar_id,
            eventId=event_id,
        ).execute()
        return _format_event(event, tz, show_full=True)
    except HttpError as e:
        if e.resp.status == 404:
            return f"Event not found: ID '{event_id}' does not exist on this calendar."
        return f"Error fetching event '{event_id}': {e}"
    finally:
        service.close()


def _search_events(query: str, max_results: int, calendar_id: str, tz: ZoneInfo) -> str:
    """Free-text search across all events (past and future)."""
    service = _get_service()
    try:
        result = service.events().list(
            calendarId=calendar_id,
            q=query,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        events = result.get("items", [])
        if not events:
            return f"No events matched: \"{query}\""

        lines = [f"Search results for \"{query}\" ({len(events)} found):"]
        for e in events:
            lines.append(_format_event(e, tz))
        return "\n\n".join(lines)
    except HttpError as e:
        return f"Error searching events: {e}"
    finally:
        service.close()


def _create_event(
    summary: str,
    start_time: str,
    end_time: str,
    description: str,
    location: str,
    attendee_emails: list[str],
    calendar_id: str,
    tz: ZoneInfo,
) -> str:
    """
    Create a new event. Returns the new event ID and a formatted summary.
    start_time / end_time: ISO 8601 strings — "2026-06-20T15:00:00"
    If no timezone offset is in the string, the user's TIMEZONE is applied.
    """
    service = _get_service()
    try:
        tz_name = str(tz)

        body: dict = {
            "summary": summary,
            "start": {"dateTime": start_time, "timeZone": tz_name},
            "end":   {"dateTime": end_time,   "timeZone": tz_name},
        }
        if description:
            body["description"] = description
        if location:
            body["location"] = location
        if attendee_emails:
            body["attendees"] = [{"email": e.strip()} for e in attendee_emails]

        created = service.events().insert(
            calendarId=calendar_id,
            body=body,
            sendUpdates="all" if attendee_emails else "none",
        ).execute()

        event_id = created.get("id", "unknown")
        start_fmt = _localize_dt_string(
            created["start"].get("dateTime") or created["start"].get("date"), tz
        )
        return (
            f"Created: \"{summary}\" at {start_fmt}\n"
            f"ID: {event_id}\n"
            f"Link: {created.get('htmlLink', '')}"
        )
    except HttpError as e:
        return f"Error creating event: {e}"
    finally:
        service.close()


def _update_event(
    event_id: str,
    query: str,
    new_summary: str,
    new_description: str,
    new_start_time: str,
    new_end_time: str,
    new_location: str,
    calendar_id: str,
    tz: ZoneInfo,
) -> str:
    """
    Patch specific fields of an existing event. Uses events().patch() so
    only the fields you pass are changed — nothing else is touched.

    If event_id is not provided, searches by query and uses the first match.
    """
    service = _get_service()
    try:
        # Resolve event_id via search if not directly provided
        if not event_id:
            if not query:
                return "Error: provide either 'event_id' or 'query' to identify the event."
            result = service.events().list(
                calendarId=calendar_id, q=query, maxResults=3, singleEvents=True
            ).execute()
            items = result.get("items", [])
            if not items:
                return f"No events found matching: \"{query}\""
            if len(items) > 1:
                # Show the matches so the LLM/user can confirm the right one
                lines = [
                    f"Found {len(items)} events matching \"{query}\". "
                    "Updating the first match. Provide 'event_id' for a specific one:"
                ]
                for e in items:
                    lines.append(_format_event(e, tz))
                lines.append("")
                event_id = items[0]["id"]
                # Still proceed with update but surface the ambiguity
                prefix = "\n".join(lines) + "\n"
            else:
                event_id = items[0]["id"]
                prefix = ""
        else:
            prefix = ""

        # Build patch body — only include fields the caller provided
        patch: dict = {}
        tz_name = str(tz)

        if new_summary:
            patch["summary"] = new_summary
        if new_description:
            patch["description"] = new_description
        if new_location:
            patch["location"] = new_location
        if new_start_time:
            patch["start"] = {"dateTime": new_start_time, "timeZone": tz_name}
        if new_end_time:
            patch["end"] = {"dateTime": new_end_time, "timeZone": tz_name}

        if not patch:
            return "Nothing to update — no new fields were provided."

        updated = service.events().patch(
            calendarId=calendar_id,
            eventId=event_id,
            body=patch,
        ).execute()

        updated_title = updated.get("summary", "(untitled)")
        start_fmt = _localize_dt_string(
            updated["start"].get("dateTime") or updated["start"].get("date"), tz
        )
        changed = ", ".join(patch.keys())
        return (
            f"{prefix}"
            f"Updated: \"{updated_title}\" (ID: {event_id})\n"
            f"Fields changed: {changed}\n"
            f"New start: {start_fmt}"
        )
    except HttpError as e:
        if e.resp.status == 404:
            return f"Event not found: ID '{event_id}' does not exist."
        return f"Error updating event: {e}"
    finally:
        service.close()


def _delete_event(event_id: str, query: str, calendar_id: str, tz: ZoneInfo) -> str:
    """
    Delete an event by event_id, or by searching with query.

    When using query:
        - If exactly one match: deletes it and confirms.
        - If multiple matches: shows the list and deletes the first one,
          telling the LLM to use event_id for a specific one next time.
        - If no match: says so without deleting anything.
    """
    service = _get_service()
    try:
        if not event_id:
            if not query:
                return "Error: provide either 'event_id' or 'query' to identify the event."

            result = service.events().list(
                calendarId=calendar_id, q=query, maxResults=5, singleEvents=True
            ).execute()
            items = result.get("items", [])

            if not items:
                return f"No events found matching: \"{query}\" — nothing deleted."

            if len(items) > 1:
                lines = [
                    f"Found {len(items)} events matching \"{query}\". "
                    "Deleted the first match. Use 'event_id' to target a specific one:"
                ]
                for e in items:
                    lines.append(_format_event(e, tz))
                event_id  = items[0]["id"]
                title     = items[0].get("summary", "(untitled)")
                service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
                lines.append(f"\nDeleted: \"{title}\" (ID: {event_id})")
                return "\n\n".join(lines)

            # Exactly one match
            event_id = items[0]["id"]
            title    = items[0].get("summary", "(untitled)")

        else:
            # Fetch title for confirmation message before deleting
            try:
                event = service.events().get(
                    calendarId=calendar_id, eventId=event_id
                ).execute()
                title = event.get("summary", "(untitled)")
            except HttpError:
                title = event_id   # fall back to ID if fetch fails

        service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
        return f"Deleted: \"{title}\" (ID: {event_id})"

    except HttpError as e:
        if e.resp.status == 404:
            return f"Event not found: ID '{event_id}' does not exist — nothing deleted."
        if e.resp.status == 410:
            return f"Event '{event_id}' was already deleted."
        return f"Error deleting event: {e}"
    finally:
        service.close()


def _list_calendars() -> str:
    """List all calendars the user has access to, with their IDs."""
    service = _get_service()
    try:
        result = service.calendarList().list().execute()
        items  = result.get("items", [])
        if not items:
            return "No calendars found on this account."

        lines = [f"Calendars ({len(items)} found):"]
        for cal in items:
            cal_id    = cal.get("id", "unknown")
            name      = cal.get("summary", "(unnamed)")
            role      = cal.get("accessRole", "unknown")
            primary   = " [PRIMARY]" if cal.get("primary") else ""
            lines.append(f"- {name}{primary}\n  ID: {cal_id}\n  Access: {role}")

        lines.append(
            "\nTo use a specific calendar, set GOOGLE_CALENDAR_ID in .env, "
            "or pass 'calendar_id' in your tool call."
        )
        return "\n\n".join(lines)
    except HttpError as e:
        return f"Error listing calendars: {e}"
    finally:
        service.close()


# ── public async entrypoint ───────────────────────────────────────────────────

async def google_calendar_action(
    action: str,
    max_results: int = 10,
    query: str = "",
    event_id: str = "",
    calendar_id: str = "",
    summary: str = "",
    description: str = "",
    location: str = "",
    start_time: str = "",
    end_time: str = "",
    attendee_emails: list[str] | None = None,
    new_summary: str = "",
    new_description: str = "",
    new_start_time: str = "",
    new_end_time: str = "",
    new_location: str = "",
) -> str:
    """
    Unified async entrypoint for all Google Calendar operations.

    Args:
        action:           which operation to perform (see actions below).
        max_results:      max events to return for list/search (1–50, default 10).
        query:            free-text search string for search/delete/update.
        event_id:         Google Calendar event ID. Get from list_events or search_events output.
                          Preferred over 'query' for delete/update — avoids ambiguity.
        calendar_id:      calendar to operate on. Defaults to GOOGLE_CALENDAR_ID env var
                          or "primary". Get IDs from list_calendars.
        summary:          event title. Required for create_event.
        description:      event body/notes. Optional for create_event.
        location:         event location string. Optional for create/update.
        start_time:       ISO 8601 datetime for event start. Required for create_event.
                          Example: "2026-06-20T15:00:00"
        end_time:         ISO 8601 datetime for event end. Required for create_event.
        attendee_emails:  list of email addresses to invite. Optional for create_event.
                          Invite emails are sent automatically when provided.
        new_summary:      replacement title for update_event.
        new_description:  replacement description for update_event.
        new_start_time:   replacement start time for update_event.
        new_end_time:     replacement end time for update_event.
        new_location:     replacement location for update_event.

    Actions:
        list_events      upcoming events, soonest first.
        get_event        full details of one event (notes, attendees, meet link).
        search_events    free-text search across all events.
        create_event     create a new event; returns new event ID.
        update_event     patch specific fields without touching others.
        delete_event     delete by event_id or keyword search.
        list_calendars   list all calendars and their IDs.

    Returns a plain string in all cases. Never raises.
    """
    max_results  = max(1, min(50, max_results))
    tz           = _get_timezone()
    cal_id       = calendar_id.strip() or _get_calendar_id()

    dispatch = {
        "list_events":   lambda: _list_events(max_results, cal_id, tz),
        "get_event":     lambda: _get_event(event_id, cal_id, tz),
        "search_events": lambda: _search_events(query, max_results, cal_id, tz),
        "create_event":  lambda: _create_event(
            summary, start_time, end_time, description,
            location, attendee_emails or [], cal_id, tz,
        ),
        "update_event":  lambda: _update_event(
            event_id, query, new_summary, new_description,
            new_start_time, new_end_time, new_location, cal_id, tz,
        ),
        "delete_event":  lambda: _delete_event(event_id, query, cal_id, tz),
        "list_calendars": lambda: _list_calendars(),
    }

    fn = dispatch.get(action)
    if fn is None:
        return (
            f"Error: unknown action '{action}'. "
            f"Valid actions: {', '.join(dispatch.keys())}."
        )

    # Validate required args before hitting the API
    if action == "get_event" and not event_id.strip():
        return "Error: 'get_event' requires an 'event_id'. Get IDs from list_events or search_events."
    if action == "search_events" and not query.strip():
        return "Error: 'search_events' requires a non-empty 'query'."
    if action == "create_event":
        missing = [f for f, v in [("summary", summary), ("start_time", start_time), ("end_time", end_time)] if not v.strip()]
        if missing:
            return f"Error: 'create_event' requires: {', '.join(missing)}."
    if action == "delete_event" and not event_id.strip() and not query.strip():
        return "Error: 'delete_event' requires either 'event_id' or 'query'."
    if action == "update_event" and not event_id.strip() and not query.strip():
        return "Error: 'update_event' requires either 'event_id' or 'query'."

    try:
        return await asyncio.to_thread(fn)
    except FileNotFoundError as e:
        return f"Setup error: {e}"
    except Exception as e:
        logger.exception("google_calendar_action failed (action=%r): %s", action, e)
        return f"Unexpected error in google_calendar '{action}': {e}"