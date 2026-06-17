"""
Google Calendar Tool — allows KAIROS to read, write, search, update, and delete calendar events.
"""

import datetime
import os
import logging
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar"]

def _get_calendar_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("credentials.json"):
                raise FileNotFoundError("Missing 'credentials.json' in root directory.")
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)


async def google_calendar_action(action: str, **kwargs) -> str:
    """Unified safe entrypoint for all Google Calendar operations."""
    try:
        import asyncio
        return await asyncio.to_thread(_execute_action, action, kwargs)
    except Exception as e:
        logger.exception("Google Calendar action failed")
        return f"Error executing calendar action '{action}': {str(e)}"


def _execute_action(action: str, args: dict) -> str:
    service = _get_calendar_service()

    # ── ACTION: LIST EVENTS ──────────────────────────────────────────────────
    if action == "list_events":
        max_results = args.get("max_results", 10)
        now = datetime.datetime.utcnow().isoformat() + "Z"
        events_result = service.events().list(
            calendarId="primary", timeMin=now, maxResults=max_results,
            singleEvents=True, orderBy="startTime"
        ).execute()
        events = events_result.get("items", [])
        if not events: return "No upcoming events found."
        
        output = ["Upcoming Calendar Events:"]
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            output.append(f"- ID: {event['id']} | [{start}] {event.get('summary', 'No Title')}")
        return "\n".join(output)

    # ── ACTION: SEARCH EVENTS (Free-text search) ──────────────────────────────
    elif action == "search_events":
        query = args.get("query", "")
        if not query:
            return "Error: A search 'query' string is required."
        
        # Look across past & future events matching the keyword string
        events_result = service.events().list(
            calendarId="primary", q=query, maxResults=5, singleEvents=True
        ).execute()
        events = events_result.get("items", [])
        if not events: 
            return f"No events matched your search query: '{query}'"
        
        output = [f"Search results for query '{query}':"]
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            output.append(f"- ID: {event['id']} | [{start}] {event.get('summary', 'No Title')}")
        return "\n".join(output)

    # ── ACTION: CREATE EVENT ──────────────────────────────────────────────────
    elif action == "create_event":
        event_body = {
            "summary": args.get("summary"),
            "description": args.get("description", ""),
            "start": {"dateTime": args.get("start_time"), "timeZone": "UTC"},
            "end": {"dateTime": args.get("end_time"), "timeZone": "UTC"},
        }
        created_event = service.events().insert(calendarId="primary", body=event_body).execute()
        return f"Success: Event '{args.get('summary')}' created."

    # ── ACTION: DELETE EVENT ──────────────────────────────────────────────────
    elif action == "delete_event":
        event_id = args.get("event_id")
        query = args.get("query")

        # If LLM didn't pass an ID but gave a name, search for the ID dynamically
        if not event_id and query:
            search_res = service.events().list(calendarId="primary", q=query, maxResults=1).execute()
            items = search_res.get("items", [])
            if not items: return f"Error: Could not find an event matching '{query}' to delete."
            event_id = items[0]["id"]
            query = items[0].get("summary") # refine label for logging

        if not event_id:
            return "Error: You must provide either an 'event_id' or a search 'query' to target a deletion."

        service.events().delete(calendarId="primary", eventId=event_id).execute()
        return f"Success: Deleted event '{event_id if not query else query}'."

    # ── ACTION: UPDATE EVENT ──────────────────────────────────────────────────
    elif action == "update_event":
        event_id = args.get("event_id")
        query = args.get("query")

        if not event_id and query:
            search_res = service.events().list(calendarId="primary", q=query, maxResults=1).execute()
            items = search_res.get("items", [])
            if not items: return f"Error: Could not find an event matching '{query}' to update."
            event_id = items[0]["id"]

        if not event_id:
            return "Error: 'event_id' or 'query' text lookup is required to update an event."

        # Fetch original event state to patch onto it
        event = service.events().get(calendarId="primary", eventId=event_id).execute()
        
        # Conditionally update fields if passed by the LLM
        if args.get("new_summary"): event["summary"] = args.get("new_summary")
        if args.get("new_description"): event["description"] = args.get("new_description")
        if args.get("new_start_time"): event["start"]["dateTime"] = args.get("new_start_time")
        if args.get("new_end_time"): event["end"]["dateTime"] = args.get("new_end_time")

        updated = service.events().update(calendarId="primary", eventId=event_id, body=event).execute()
        return f"Success: Updated event '{updated.get('summary')}'."

    return f"Error: Unknown calendar action '{action}'"