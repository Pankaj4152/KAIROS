"""
Gmail Check Tool — reads inbox via IMAP, with multi-action support.

Actions:
    list_unread     — fetch headers of recent unread emails (default)
    list_recent     — fetch headers of recent emails regardless of read status
    search          — search inbox by keyword (subject, sender, or body snippet)
    get_body        — fetch the text body of a specific email by UID
    mark_read       — mark a specific email as read by UID
    count_unread    — return unread count only (fast, low-bandwidth)

Security:
    Requires GMAIL_USER + GMAIL_APP_PASSWORD (Google App Password, not master password).
    IMAP access must be enabled in Gmail settings.
    No OAuth involved — straight IMAP.

Robustness:
    - Connection is always closed in a finally block (no leaks).
    - Decodes RFC 2047-encoded headers safely (handles UTF-8 and other charsets).
    - Strips HTML from email bodies, returns plain text only.
    - Every action has a defined failure string — never raises to the LLM.
    - UID-based operations are stable across mailbox reindexing.

Env vars required:
    GMAIL_USER           your Gmail address
    GMAIL_APP_PASSWORD   16-char Google App Password (not your account password)
"""

import asyncio
import email
import imaplib
import logging
import os
import re
from email.header import decode_header as _decode_header
from html.parser import HTMLParser

logger = logging.getLogger(__name__)

# ── connection ────────────────────────────────────────────────────────────────

def _get_credentials() -> tuple[str, str] | tuple[None, None]:
    """Return (username, password) or (None, None) if env vars are missing."""
    username = os.getenv("GMAIL_USER", "").strip()
    password = os.getenv("GMAIL_APP_PASSWORD", "").strip()
    if not username or not password:
        return None, None
    return username, password


def _connect() -> imaplib.IMAP4_SSL | None:
    """
    Open an IMAP SSL connection and log in.
    Returns the connection object or None on failure.
    Caller is responsible for calling conn.logout() in a finally block.
    """
    username, password = _get_credentials()
    if not username:
        return None
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com", timeout=10)
        mail.login(username, password)
        return mail
    except imaplib.IMAP4.error as e:
        logger.warning("Gmail IMAP login failed: %s", e)
        return None
    except OSError as e:
        logger.warning("Gmail IMAP connection failed (network): %s", e)
        return None


# ── header decoding ───────────────────────────────────────────────────────────

def _decode_header_value(raw_value: str | None) -> str:
    """
    Decode a potentially RFC 2047-encoded header value (Subject, From, etc.).

    RFC 2047 encodes non-ASCII headers as =?charset?encoding?...?=.
    Python's email.header.decode_header handles this, but returns a list
    of (bytes, charset) tuples that must be reassembled.
    """
    if not raw_value:
        return ""
    parts = []
    for fragment, charset in _decode_header(raw_value):
        if isinstance(fragment, bytes):
            try:
                parts.append(fragment.decode(charset or "utf-8", errors="replace"))
            except (LookupError, UnicodeDecodeError):
                parts.append(fragment.decode("utf-8", errors="replace"))
        else:
            parts.append(fragment)
    return " ".join(parts).strip()


# ── HTML stripping ────────────────────────────────────────────────────────────

class _HTMLStripper(HTMLParser):
    """Minimal HTML-to-text converter for email body extraction."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts).strip()


def _strip_html(html: str) -> str:
    stripper = _HTMLStripper()
    stripper.feed(html)
    text = stripper.get_text()
    # Collapse whitespace runs
    return re.sub(r"\s{3,}", "\n\n", text).strip()


# ── body extraction ───────────────────────────────────────────────────────────

def _extract_body(msg: email.message.Message, max_chars: int = 1500) -> str:
    """
    Walk a MIME message and return the best readable body text.

    Priority:
        1. text/plain parts
        2. text/html parts (stripped to plain text)

    Truncates at max_chars to stay within prompt budget.
    """
    plain_parts: list[str] = []
    html_parts: list[str] = []

    for part in msg.walk():
        content_type = part.get_content_type()
        if part.get("Content-Disposition", "").startswith("attachment"):
            continue
        charset = part.get_content_charset() or "utf-8"

        if content_type == "text/plain":
            payload = part.get_payload(decode=True)
            if payload:
                try:
                    plain_parts.append(payload.decode(charset, errors="replace"))
                except (LookupError, UnicodeDecodeError):
                    plain_parts.append(payload.decode("utf-8", errors="replace"))

        elif content_type == "text/html" and not plain_parts:
            payload = part.get_payload(decode=True)
            if payload:
                try:
                    html_parts.append(_strip_html(payload.decode(charset, errors="replace")))
                except (LookupError, UnicodeDecodeError):
                    html_parts.append(_strip_html(payload.decode("utf-8", errors="replace")))

    body = "\n\n".join(plain_parts or html_parts).strip()
    if len(body) > max_chars:
        body = body[:max_chars] + f"\n\n[… truncated at {max_chars} chars]"
    return body or "(no readable body)"


# ── UID fetching helpers ──────────────────────────────────────────────────────

def _fetch_uids(mail: imaplib.IMAP4_SSL, criteria: str) -> list[bytes]:
    """
    Search the selected mailbox by criteria string and return UID list.
    Uses UID SEARCH for stable identifiers across sessions.
    """
    status, data = mail.uid("search", None, criteria)
    if status != "OK" or not data or data[0] is None:
        return []
    raw = data[0].strip()
    if not raw:
        return []
    return raw.split()


def _fetch_header_msg(mail: imaplib.IMAP4_SSL, uid: bytes) -> email.message.Message | None:
    """Fetch and parse header-only for a single UID."""
    status, data = mail.uid("fetch", uid, "(RFC822.HEADER)")
    if status != "OK" or not data or data[0] is None:
        return None
    raw = data[0][1]
    if not isinstance(raw, bytes):
        return None
    return email.message_from_bytes(raw)


def _fetch_full_msg(mail: imaplib.IMAP4_SSL, uid: bytes) -> email.message.Message | None:
    """Fetch and parse the full message (headers + body) for a single UID."""
    status, data = mail.uid("fetch", uid, "(RFC822)")
    if status != "OK" or not data or data[0] is None:
        return None
    raw = data[0][1]
    if not isinstance(raw, bytes):
        return None
    return email.message_from_bytes(raw)


def _format_header(count: int, uid: bytes, msg: email.message.Message) -> str:
    """Format one email header entry for LLM consumption."""
    subject = _decode_header_value(msg.get("Subject"))
    sender  = _decode_header_value(msg.get("From"))
    date    = msg.get("Date", "Unknown Date")
    return (
        f"{count}. UID: {uid.decode()}\n"
        f"   FROM: {sender}\n"
        f"   SUBJECT: {subject}\n"
        f"   DATE: {date}"
    )


# ── actions (all sync — run inside asyncio.to_thread) ────────────────────────

def _count_unread() -> str:
    """Fast unread count — fetches no message data, just the search count."""
    mail = _connect()
    if mail is None:
        return "Error: Gmail credentials missing or IMAP login failed."
    try:
        mail.select("inbox", readonly=True)
        uids = _fetch_uids(mail, "UNSEEN")
        return f"You have {len(uids)} unread email{'s' if len(uids) != 1 else ''} in your inbox."
    except Exception as e:
        logger.warning("count_unread failed: %s", e)
        return f"Failed to count unread emails: {e}"
    finally:
        try:
            mail.logout()
        except Exception:
            pass


def _list_headers(criteria: str, max_emails: int, label: str) -> str:
    """
    Core header listing logic, shared by list_unread and list_recent.
    criteria: IMAP search string, e.g. 'UNSEEN' or 'ALL'
    """
    mail = _connect()
    if mail is None:
        return "Error: Gmail credentials missing or IMAP login failed."
    try:
        mail.select("inbox", readonly=True)
        uids = _fetch_uids(mail, criteria)

        if not uids:
            return f"No {label} emails found in your inbox."

        total = len(uids)
        # Most recent first (IMAP returns oldest first by default)
        uids = list(reversed(uids))[:max_emails]

        lines = [f"{label.capitalize()} emails (showing {len(uids)} of {total}):"]
        for i, uid in enumerate(uids, 1):
            msg = _fetch_header_msg(mail, uid)
            if msg is None:
                lines.append(f"{i}. UID: {uid.decode()} — (could not fetch header)")
                continue
            lines.append(_format_header(i, uid, msg))

        return "\n\n".join(lines)

    except Exception as e:
        logger.warning("list_headers failed (criteria=%r): %s", criteria, e)
        return f"Failed to retrieve emails: {e}"
    finally:
        try:
            mail.logout()
        except Exception:
            pass


def _search_emails(query: str, max_emails: int) -> str:
    """
    Search inbox by keyword. Searches subject + body (Gmail IMAP TEXT search).
    Returns header summaries of matching emails with their UIDs.
    """
    mail = _connect()
    if mail is None:
        return "Error: Gmail credentials missing or IMAP login failed."
    try:
        mail.select("inbox", readonly=True)
        # TEXT searches subject, body, and headers. Use quoted string for phrase search.
        safe_query = query.replace('"', "")          # prevent IMAP injection
        criteria   = f'TEXT "{safe_query}"'
        uids = _fetch_uids(mail, criteria)

        if not uids:
            return f"No emails matched your search for: \"{query}\""

        total = len(uids)
        uids  = list(reversed(uids))[:max_emails]

        lines = [f"Search results for \"{query}\" ({len(uids)} of {total} shown):"]
        for i, uid in enumerate(uids, 1):
            msg = _fetch_header_msg(mail, uid)
            if msg is None:
                lines.append(f"{i}. UID: {uid.decode()} — (could not fetch header)")
                continue
            lines.append(_format_header(i, uid, msg))

        return "\n\n".join(lines)

    except Exception as e:
        logger.warning("search_emails failed (query=%r): %s", query, e)
        return f"Failed to search emails: {e}"
    finally:
        try:
            mail.logout()
        except Exception:
            pass


def _get_body(uid_str: str) -> str:
    """
    Fetch the full body of a specific email by UID.
    uid_str comes from a previous list/search call — shown in output as 'UID: ...'.
    """
    mail = _connect()
    if mail is None:
        return "Error: Gmail credentials missing or IMAP login failed."
    try:
        mail.select("inbox", readonly=True)
        uid = uid_str.strip().encode()
        msg = _fetch_full_msg(mail, uid)

        if msg is None:
            return f"Could not fetch email with UID {uid_str}. It may no longer exist."

        subject = _decode_header_value(msg.get("Subject"))
        sender  = _decode_header_value(msg.get("From"))
        date    = msg.get("Date", "Unknown Date")
        body    = _extract_body(msg)

        return (
            f"FROM: {sender}\n"
            f"SUBJECT: {subject}\n"
            f"DATE: {date}\n\n"
            f"BODY:\n{body}"
        )

    except Exception as e:
        logger.warning("get_body failed (uid=%r): %s", uid_str, e)
        return f"Failed to fetch email body: {e}"
    finally:
        try:
            mail.logout()
        except Exception:
            pass


def _mark_read(uid_str: str) -> str:
    """
    Mark a specific email as read (sets the \\Seen flag) by UID.
    Opens mailbox in read-write mode for this action only.
    """
    mail = _connect()
    if mail is None:
        return "Error: Gmail credentials missing or IMAP login failed."
    try:
        mail.select("inbox")          # read-write (no readonly=True)
        uid = uid_str.strip().encode()
        status, data = mail.uid("store", uid, "+FLAGS", "\\Seen")

        if status != "OK":
            return f"Failed to mark email {uid_str} as read (IMAP status: {status})."

        return f"Marked email UID {uid_str} as read."

    except Exception as e:
        logger.warning("mark_read failed (uid=%r): %s", uid_str, e)
        return f"Failed to mark email as read: {e}"
    finally:
        try:
            mail.logout()
        except Exception:
            pass


# ── public async entrypoint ───────────────────────────────────────────────────

async def check_gmail(
    action: str = "list_unread",
    max_results: int = 5,
    query: str = "",
    uid: str = "",
) -> str:
    """
    Unified async entrypoint for all Gmail IMAP operations.

    Args:
        action:      which operation to perform (see below).
        max_results: max emails to return for list/search actions (1–20).
        query:       search keyword for the 'search' action.
        uid:         email UID string for 'get_body' and 'mark_read' actions.
                     UIDs appear in the output of list_unread/list_recent/search
                     as "UID: <value>" — copy the value to use here.

    Actions:
        list_unread  — headers of recent unread emails (default).
        list_recent  — headers of recent emails, read or unread.
        search       — search inbox by keyword (requires `query`).
        get_body     — full body of one email (requires `uid`).
        mark_read    — mark one email as read (requires `uid`).
        count_unread — unread count only, no headers.

    Returns a plain string in all cases. Never raises.
    """
    # Clamp max_results to a sane range
    max_results = max(1, min(20, max_results))

    dispatch = {
        "list_unread":  lambda: _list_headers("UNSEEN", max_results, "unread"),
        "list_recent":  lambda: _list_headers("ALL",    max_results, "recent"),
        "search":       lambda: _search_emails(query, max_results),
        "get_body":     lambda: _get_body(uid),
        "mark_read":    lambda: _mark_read(uid),
        "count_unread": lambda: _count_unread(),
    }

    fn = dispatch.get(action)
    if fn is None:
        return (
            f"Error: unknown action '{action}'. "
            f"Valid actions: {', '.join(dispatch.keys())}."
        )

    # Validate required args before hitting the network
    if action == "search" and not query.strip():
        return "Error: 'search' action requires a non-empty 'query' parameter."
    if action in ("get_body", "mark_read") and not uid.strip():
        return f"Error: '{action}' action requires a 'uid' parameter. Get UIDs from list_unread or search."

    return await asyncio.to_thread(fn)