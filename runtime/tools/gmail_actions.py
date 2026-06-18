"""
Gmail Actions Tool — send, reply, forward, delete, move, archive, and manage emails.

This is the write-side companion to gmail_check.py (which handles read-only IMAP).
Both tools share the same env vars and the same _connect() / _get_credentials() pattern.

Actions:
    send            — compose and send a new email (To, CC, BCC, subject, body)
    reply           — reply to an email by UID (fetches original for threading headers)
    reply_all       — reply to all recipients of an email by UID
    forward         — forward an email by UID to new recipients
    delete          — move email to Trash by UID (permanent delete requires expunge)
    delete_permanent— permanently delete email by UID (no Trash, unrecoverable)
    archive         — move email to All Mail (removes from Inbox, keeps in Gmail)
    move            — move email to a named folder/label by UID
    mark_unread     — mark a read email as unread by UID
    list_folders    — list all mailbox folders/labels available on the account
    create_draft    — save a draft to [Gmail]/Drafts via IMAP APPEND

Split from gmail_check.py deliberately:
    - Executor can set different permission levels per tool (read vs write)
    - Registry can disable write actions independently of read actions
    - Tests are cleanly separated (read mocks vs send mocks)
    - LLM gets tighter tool descriptions that match actual capability

Auth:
    SMTP (send/reply/forward/draft):
        GMAIL_USER + GMAIL_APP_PASSWORD
        Port 587 + STARTTLS (same as channels/email.py)

    IMAP (delete/move/archive/mark_unread/list_folders):
        GMAIL_USER + GMAIL_APP_PASSWORD
        imap.gmail.com:993 SSL
        IMAP must be enabled in Gmail settings.

    Both use the same App Password — no separate credentials needed.

Design rules (matching gmail_check.py):
    - Every action returns a plain string. Never raises to the LLM.
    - Connections always closed in finally blocks.
    - Addresses validated before any network call.
    - UID-based IMAP operations (stable across reindexing).
    - All blocking I/O offloaded via asyncio.to_thread().
"""

import asyncio
import email as email_lib
import email.utils
import imaplib
import logging
import os
import re
import smtplib
import time
from email.header import decode_header as _decode_header
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

# ── Gmail IMAP folder names ────────────────────────────────────────────────────
# These are fixed across all Gmail accounts.
# list_folders() will return the real names if the account is non-English.
GMAIL_FOLDERS = {
    "inbox":    "INBOX",
    "sent":     "[Gmail]/Sent Mail",
    "drafts":   "[Gmail]/Drafts",
    "spam":     "[Gmail]/Spam",
    "trash":    "[Gmail]/Trash",
    "archive":  "[Gmail]/All Mail",
    "starred":  "[Gmail]/Starred",
    "important":"[Gmail]/Important",
}

# ── validation ────────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)


def _validate_address(addr: str) -> str | None:
    """
    Return the bare email address if valid, else None.
    Accepts both "user@domain.com" and "Name <user@domain.com>" formats.
    """
    addr = addr.strip()
    # Strip display name if present
    _, bare = email.utils.parseaddr(addr)
    if not bare:
        return None
    if not _EMAIL_RE.match(bare):
        return None
    return bare


def _parse_address_list(raw: str) -> tuple[list[str], list[str]]:
    """
    Parse a comma-separated address string into (valid, invalid) lists.
    Returns bare email addresses in valid list.
    """
    if not raw or not raw.strip():
        return [], []
    items = [item.strip() for item in raw.split(",") if item.strip()]
    valid, invalid = [], []
    for item in items:
        parsed = email.utils.parseaddr(item)
        addr = parsed[1].strip() if parsed[1] else ""
        if not addr and "@" in item:
            addr = item.strip()
        if addr and _EMAIL_RE.match(addr):
            valid.append(addr)
        else:
            invalid.append(item)
    return valid, invalid


# ── credentials ───────────────────────────────────────────────────────────────

def _get_credentials() -> tuple[str, str] | tuple[None, None]:
    """Return (username, password) or (None, None) if env vars are missing."""
    username = os.getenv("GMAIL_USER", "").strip()
    password = os.getenv("GMAIL_APP_PASSWORD", "").strip()
    if not username or not password:
        return None, None
    return username, password


# ── IMAP connection ───────────────────────────────────────────────────────────

def _imap_connect() -> imaplib.IMAP4_SSL | None:
    """
    Open an IMAP SSL connection and log in.
    Returns the connection or None on failure.
    Caller must call .logout() in a finally block.
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
        logger.warning("Gmail IMAP connection failed: %s", e)
        return None


# ── SMTP connection ───────────────────────────────────────────────────────────

def _smtp_connect() -> smtplib.SMTP | None:
    """
    Open an SMTP connection with STARTTLS.
    Returns the connection or None on failure.
    Caller must call .quit() in a finally block.
    """
    username, password = _get_credentials()
    if not username:
        return None
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=10)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(username, password)
        return server
    except smtplib.SMTPException as e:
        logger.warning("Gmail SMTP login failed: %s", e)
        return None
    except OSError as e:
        logger.warning("Gmail SMTP connection failed: %s", e)
        return None


# ── header decoding (shared with gmail_check pattern) ─────────────────────────

def _decode_header_value(raw_value: str | None) -> str:
    """Decode RFC 2047-encoded header values safely."""
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


# ── message building ──────────────────────────────────────────────────────────

def _build_message(
    from_addr: str,
    to: list[str],
    subject: str,
    body: str,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
    reply_to_message_id: str | None = None,
    references: str | None = None,
    extra_headers: dict | None = None,
) -> MIMEMultipart:
    """
    Build a MIMEMultipart email message.

    reply_to_message_id: Message-ID of the email being replied to.
                         Sets In-Reply-To and updates References for threading.
    references:          Existing References header from the original email.
    extra_headers:       Any additional headers to set (e.g. X-Forwarded-To).
    """
    msg = MIMEMultipart("alternative")
    msg["From"]    = from_addr
    msg["To"]      = ", ".join(to)
    msg["Subject"] = subject
    msg["Date"]    = email.utils.formatdate(localtime=True)
    msg["Message-ID"] = email.utils.make_msgid(domain=from_addr.split("@")[-1])

    if cc:
        msg["Cc"] = ", ".join(cc)
    if bcc:
        # BCC is NOT added as a header — it's passed to sendmail() recipient list only
        pass

    if reply_to_message_id:
        msg["In-Reply-To"] = reply_to_message_id
        # References: append new message ID to existing chain
        ref_chain = f"{references} {reply_to_message_id}".strip() if references else reply_to_message_id
        msg["References"] = ref_chain

    if extra_headers:
        for k, v in extra_headers.items():
            msg[k] = v

    # Plain text part
    try:
        body.encode("us-ascii")
        msg.attach(MIMEText(body, "plain"))
    except UnicodeEncodeError:
        msg.attach(MIMEText(body, "plain", "utf-8"))

    return msg


def _fetch_original_for_reply(
    mail: imaplib.IMAP4_SSL, uid: bytes
) -> dict:
    """
    Fetch headers needed to construct a reply or forward.

    Returns dict with keys: message_id, references, subject, from_, to, cc, body_snippet
    All values are strings (empty string if not found).
    """
    result = {
        "message_id": "",
        "references": "",
        "subject": "",
        "from_": "",
        "to": "",
        "cc": "",
        "body_snippet": "",
    }

    status, data = mail.uid("fetch", uid, "(RFC822)")
    if status != "OK" or not data or data[0] is None:
        return result

    raw = data[0][1]
    if not isinstance(raw, bytes):
        return result

    msg = email_lib.message_from_bytes(raw)
    result["message_id"] = msg.get("Message-ID", "").strip()
    result["references"]  = msg.get("References", "").strip()
    result["subject"]     = _decode_header_value(msg.get("Subject", ""))
    result["from_"]       = _decode_header_value(msg.get("From", ""))
    result["to"]          = _decode_header_value(msg.get("To", ""))
    result["cc"]          = _decode_header_value(msg.get("Cc", ""))

    # Extract a short body snippet for forward quoting
    for part in msg.walk():
        if part.get_content_type() == "text/plain" and not part.get("Content-Disposition", "").startswith("attachment"):
            payload = part.get_payload(decode=True)
            if payload:
                charset = part.get_content_charset() or "utf-8"
                try:
                    text = payload.decode(charset, errors="replace")
                except (LookupError, UnicodeDecodeError):
                    text = payload.decode("utf-8", errors="replace")
                result["body_snippet"] = text[:2000]
                break

    return result


# ── SMTP actions ──────────────────────────────────────────────────────────────

def _send_email(
    to: str,
    subject: str,
    body: str,
    cc: str,
    bcc: str,
) -> str:
    """
    Compose and send a new email via SMTP.
    to/cc/bcc: comma-separated address strings.
    """
    username, _ = _get_credentials()
    if not username:
        return "Error: GMAIL_USER or GMAIL_APP_PASSWORD not set in environment."

    # Parse and validate recipients
    to_valid, to_invalid = _parse_address_list(to)
    if not to_valid:
        return f"Error: No valid 'to' addresses found. Invalid: {to or '(empty)'}."

    cc_valid, cc_invalid  = _parse_address_list(cc)
    bcc_valid, bcc_invalid = _parse_address_list(bcc)

    invalid_all = to_invalid + cc_invalid + bcc_invalid
    if invalid_all:
        return f"Error: Invalid email addresses found: {', '.join(invalid_all)}. Fix them and retry."

    if not subject.strip():
        return "Error: 'subject' must not be empty."
    if not body.strip():
        return "Error: 'body' must not be empty."

    all_recipients = to_valid + cc_valid + bcc_valid

    msg = _build_message(
        from_addr=username,
        to=to_valid,
        subject=subject,
        body=body,
        cc=cc_valid or None,
        bcc=bcc_valid or None,
    )

    server = _smtp_connect()
    if server is None:
        return "Error: Gmail SMTP login failed. Check GMAIL_USER and GMAIL_APP_PASSWORD."
    try:
        server.sendmail(username, all_recipients, msg.as_string())
        cc_note = f" (CC: {', '.join(cc_valid)})" if cc_valid else ""
        bcc_note = f" (BCC: {len(bcc_valid)} recipient(s))" if bcc_valid else ""
        return (
            f"Sent: \"{subject}\" → {', '.join(to_valid)}{cc_note}{bcc_note}"
        )
    except smtplib.SMTPRecipientsRefused as e:
        refused = ", ".join(str(k) for k in e.recipients.keys())
        return f"Error: Recipients refused by Gmail: {refused}."
    except smtplib.SMTPException as e:
        logger.warning("SMTP send failed: %s", e)
        return f"Error sending email: {e}"
    finally:
        try:
            server.quit()
        except Exception:
            pass


def _reply_email(
    uid: str,
    body: str,
    reply_all: bool,
    extra_to: str,
) -> str:
    """
    Reply to an existing email identified by IMAP UID.

    reply_all=True: reply to From + all To + all CC recipients.
    extra_to: additional addresses to include (comma-separated string).

    Threading is handled correctly via In-Reply-To and References headers.
    """
    username, _ = _get_credentials()
    if not username:
        return "Error: GMAIL_USER or GMAIL_APP_PASSWORD not set."

    if not uid.strip():
        return "Error: 'uid' is required for reply."
    if not body.strip():
        return "Error: 'body' must not be empty."

    # Fetch original email headers via IMAP
    mail = _imap_connect()
    if mail is None:
        return "Error: Gmail IMAP login failed."

    try:
        mail.select("INBOX", readonly=True)
        original = _fetch_original_for_reply(mail, uid.strip().encode())
    finally:
        try:
            mail.logout()
        except Exception:
            pass

    if not original["message_id"] and not original["subject"]:
        return f"Error: Could not fetch email with UID {uid}. It may not exist in INBOX."

    # Build subject
    subj = original["subject"]
    if not subj.lower().startswith("re:"):
        subj = f"Re: {subj}"

    # Build recipient list
    # Always reply to the original sender
    original_from_valid, _ = _parse_address_list(original["from_"])
    to_addresses = [a for a in original_from_valid if a.lower() != username.lower()]

    if reply_all:
        # Add all original To and CC (excluding ourselves)
        orig_to_valid,  _ = _parse_address_list(original["to"])
        orig_cc_valid,  _ = _parse_address_list(original["cc"])
        for addr in orig_to_valid + orig_cc_valid:
            if addr.lower() != username.lower() and addr not in to_addresses:
                to_addresses.append(addr)

    # Add any extra addresses the user specified
    if extra_to:
        extra_valid, extra_invalid = _parse_address_list(extra_to)
        if extra_invalid:
            return f"Error: Invalid extra_to addresses: {', '.join(extra_invalid)}."
        for addr in extra_valid:
            if addr not in to_addresses:
                to_addresses.append(addr)

    if not to_addresses:
        return "Error: Could not determine reply recipients from original email."

    msg = _build_message(
        from_addr=username,
        to=to_addresses,
        subject=subj,
        body=body,
        reply_to_message_id=original["message_id"],
        references=original["references"],
    )

    server = _smtp_connect()
    if server is None:
        return "Error: Gmail SMTP login failed."
    try:
        server.sendmail(username, to_addresses, msg.as_string())
        kind = "Reply-all" if reply_all else "Reply"
        return f"{kind} sent: \"{subj}\" → {', '.join(to_addresses)}"
    except smtplib.SMTPException as e:
        logger.warning("SMTP reply failed: %s", e)
        return f"Error sending reply: {e}"
    finally:
        try:
            server.quit()
        except Exception:
            pass


def _forward_email(uid: str, to: str, note: str) -> str:
    """
    Forward an email by UID to new recipients.

    note: optional text to prepend before the forwarded body.
    The original email body is quoted below the note.
    """
    username, _ = _get_credentials()
    if not username:
        return "Error: GMAIL_USER or GMAIL_APP_PASSWORD not set."

    if not uid.strip():
        return "Error: 'uid' is required for forward."

    to_valid, to_invalid = _parse_address_list(to)
    if not to_valid:
        return f"Error: No valid 'to' addresses. Got: {to or '(empty)'}."
    if to_invalid:
        return f"Error: Invalid addresses in 'to': {', '.join(to_invalid)}."

    # Fetch original
    mail = _imap_connect()
    if mail is None:
        return "Error: Gmail IMAP login failed."
    try:
        mail.select("INBOX", readonly=True)
        original = _fetch_original_for_reply(mail, uid.strip().encode())
    finally:
        try:
            mail.logout()
        except Exception:
            pass

    if not original["subject"]:
        return f"Error: Could not fetch email with UID {uid}."

    # Build subject
    subj = original["subject"]
    if not subj.lower().startswith("fwd:"):
        subj = f"Fwd: {subj}"

    # Build forwarded body
    separator = "-" * 40
    fwd_header = (
        f"\n\n{separator}\n"
        f"Forwarded message\n"
        f"From: {original['from_']}\n"
        f"Subject: {original['subject']}\n"
        f"{separator}\n\n"
    )
    full_body = (note.strip() + fwd_header if note.strip() else fwd_header) + original["body_snippet"]

    msg = _build_message(
        from_addr=username,
        to=to_valid,
        subject=subj,
        body=full_body,
    )

    server = _smtp_connect()
    if server is None:
        return "Error: Gmail SMTP login failed."
    try:
        server.sendmail(username, to_valid, msg.as_string())
        return f"Forwarded: \"{subj}\" → {', '.join(to_valid)}"
    except smtplib.SMTPException as e:
        logger.warning("SMTP forward failed: %s", e)
        return f"Error forwarding email: {e}"
    finally:
        try:
            server.quit()
        except Exception:
            pass


def _create_draft(to: str, subject: str, body: str, cc: str) -> str:
    """
    Save a draft to [Gmail]/Drafts via IMAP APPEND.
    The draft appears in Gmail's Drafts folder and can be edited/sent from the Gmail UI.
    """
    username, _ = _get_credentials()
    if not username:
        return "Error: GMAIL_USER or GMAIL_APP_PASSWORD not set."

    to_valid, to_invalid = _parse_address_list(to)
    if not to_valid:
        return f"Error: No valid 'to' addresses. Got: {to or '(empty)'}."
    if to_invalid:
        return f"Error: Invalid addresses: {', '.join(to_invalid)}."
    if not subject.strip():
        return "Error: 'subject' must not be empty."
    if not body.strip():
        return "Error: 'body' must not be empty."

    cc_valid, cc_invalid = _parse_address_list(cc)
    if cc_invalid:
        return f"Error: Invalid CC addresses: {', '.join(cc_invalid)}."

    msg = _build_message(
        from_addr=username,
        to=to_valid,
        subject=subject,
        body=body,
        cc=cc_valid or None,
    )

    mail = _imap_connect()
    if mail is None:
        return "Error: Gmail IMAP login failed."
    try:
        # APPEND requires the message as bytes, current time, and \\Draft flag
        import time as _time
        imap_time = imaplib.Time2Internaldate(_time.time())
        status, data = mail.append(
            "[Gmail]/Drafts",
            "\\Draft",
            imap_time,
            msg.as_bytes(),
        )
        if status != "OK":
            return f"Error: IMAP APPEND failed (status={status})."
        return f"Draft saved: \"{subject}\" → {', '.join(to_valid)}"
    except Exception as e:
        logger.warning("create_draft failed: %s", e)
        return f"Error saving draft: {e}"
    finally:
        try:
            mail.logout()
        except Exception:
            pass


# ── IMAP mutation actions ─────────────────────────────────────────────────────

def _delete_email(uid: str, permanent: bool, folder: str) -> str:
    """
    Delete an email by UID.

    permanent=False (default): moves to [Gmail]/Trash.
    permanent=True: sets \\Deleted flag and expunges immediately. UNRECOVERABLE.

    folder: which mailbox the UID lives in (default: INBOX).
    """
    username, _ = _get_credentials()
    if not username:
        return "Error: GMAIL_USER or GMAIL_APP_PASSWORD not set."
    if not uid.strip():
        return "Error: 'uid' is required."

    mail = _imap_connect()
    if mail is None:
        return "Error: Gmail IMAP login failed."
    try:
        # Resolve folder name
        imap_folder = GMAIL_FOLDERS.get(folder.lower(), folder) if folder else "INBOX"
        mail.select(imap_folder)

        uid_bytes = uid.strip().encode()

        if permanent:
            # Set \Deleted flag and expunge immediately
            status, _ = mail.uid("store", uid_bytes, "+FLAGS", "\\Deleted")
            if status != "OK":
                return f"Error: Could not flag email UID {uid} for deletion."
            mail.expunge()
            return f"Permanently deleted email UID {uid} from {imap_folder}. This is unrecoverable and cannot be undone."
        else:
            # Move to Trash: Gmail supports the MOVE extension
            # Try MOVE first (RFC 6851), fall back to COPY+STORE+EXPUNGE
            try:
                status, _ = mail.uid("MOVE", uid_bytes, "[Gmail]/Trash")
                if status == "OK":
                    return f"Moved email UID {uid} to Trash."
            except imaplib.IMAP4.error:
                pass

            # Fallback: COPY to Trash, then mark original as Deleted
            status, _ = mail.uid("copy", uid_bytes, "[Gmail]/Trash")
            if status != "OK":
                return f"Error: Could not move email UID {uid} to Trash."
            mail.uid("store", uid_bytes, "+FLAGS", "\\Deleted")
            mail.expunge()
            return f"Moved email UID {uid} to Trash."

    except imaplib.IMAP4.error as e:
        return f"Error: IMAP error during delete: {e}"
    finally:
        try:
            mail.logout()
        except Exception:
            pass


def _archive_email(uid: str, folder: str) -> str:
    """
    Archive an email — moves it to [Gmail]/All Mail, removing it from Inbox.
    The email is not deleted and remains searchable in Gmail.
    """
    username, _ = _get_credentials()
    if not username:
        return "Error: GMAIL_USER or GMAIL_APP_PASSWORD not set."
    if not uid.strip():
        return "Error: 'uid' is required."

    mail = _imap_connect()
    if mail is None:
        return "Error: Gmail IMAP login failed."
    try:
        imap_folder = GMAIL_FOLDERS.get(folder.lower(), folder) if folder else "INBOX"
        mail.select(imap_folder)
        uid_bytes = uid.strip().encode()

        # Try MOVE first
        try:
            status, _ = mail.uid("MOVE", uid_bytes, "[Gmail]/All Mail")
            if status == "OK":
                return f"Archived email UID {uid} (moved to All Mail, removed from {imap_folder})."
        except imaplib.IMAP4.error:
            pass

        # Fallback: COPY to All Mail, then delete from source
        status, _ = mail.uid("copy", uid_bytes, "[Gmail]/All Mail")
        if status != "OK":
            return f"Error: Could not archive email UID {uid}."
        mail.uid("store", uid_bytes, "+FLAGS", "\\Deleted")
        mail.expunge()
        return f"Archived email UID {uid} (moved to All Mail, removed from {imap_folder})."

    except imaplib.IMAP4.error as e:
        return f"Error: IMAP error during archive: {e}"
    finally:
        try:
            mail.logout()
        except Exception:
            pass


def _move_email(uid: str, destination: str, source_folder: str) -> str:
    """
    Move an email to a named folder.
    destination: folder short name (inbox/sent/drafts/spam/trash/archive/starred)
                 OR a full IMAP path like "[Gmail]/All Mail" or a custom label name.
    """
    username, _ = _get_credentials()
    if not username:
        return "Error: GMAIL_USER or GMAIL_APP_PASSWORD not set."
    if not uid.strip():
        return "Error: 'uid' is required."
    if not destination.strip():
        return "Error: 'destination' folder name is required."

    mail = _imap_connect()
    if mail is None:
        return "Error: Gmail IMAP login failed."
    try:
        src = GMAIL_FOLDERS.get(source_folder.lower(), source_folder) if source_folder else "INBOX"
        dst = GMAIL_FOLDERS.get(destination.lower(), destination)
        mail.select(src)
        uid_bytes = uid.strip().encode()

        # Try MOVE extension first
        try:
            status, _ = mail.uid("MOVE", uid_bytes, dst)
            if status == "OK":
                return f"Moved email UID {uid} from {src} to {dst}."
        except imaplib.IMAP4.error:
            pass

        # Fallback: COPY + mark deleted + expunge
        status, _ = mail.uid("copy", uid_bytes, dst)
        if status != "OK":
            return f"Error: Could not move email UID {uid} to '{dst}'. Check folder name."
        mail.uid("store", uid_bytes, "+FLAGS", "\\Deleted")
        mail.expunge()
        return f"Moved email UID {uid} from {src} to {dst}."

    except imaplib.IMAP4.error as e:
        return f"Error: IMAP error during move: {e}"
    finally:
        try:
            mail.logout()
        except Exception:
            pass


def _mark_unread(uid: str, folder: str) -> str:
    """Mark a read email as unread by removing the \\Seen flag."""
    username, _ = _get_credentials()
    if not username:
        return "Error: GMAIL_USER or GMAIL_APP_PASSWORD not set."
    if not uid.strip():
        return "Error: 'uid' is required."

    mail = _imap_connect()
    if mail is None:
        return "Error: Gmail IMAP login failed."
    try:
        imap_folder = GMAIL_FOLDERS.get(folder.lower(), folder) if folder else "INBOX"
        mail.select(imap_folder)
        status, _ = mail.uid("store", uid.strip().encode(), "-FLAGS", "\\Seen")
        if status != "OK":
            return f"Error: Could not mark email UID {uid} as unread."
        return f"Marked email UID {uid} as unread."
    except imaplib.IMAP4.error as e:
        return f"Error: IMAP error during mark_unread: {e}"
    finally:
        try:
            mail.logout()
        except Exception:
            pass


def _list_folders() -> str:
    """
    List all mailbox folders/labels available on this Gmail account.
    Includes both standard Gmail folders and custom labels.
    """
    username, _ = _get_credentials()
    if not username:
        return "Error: GMAIL_USER or GMAIL_APP_PASSWORD not set."

    mail = _imap_connect()
    if mail is None:
        return "Error: Gmail IMAP login failed."
    try:
        status, folder_list = mail.list()
        if status != "OK":
            return "Error: Could not retrieve folder list from Gmail."

        lines = ["Available Gmail folders/labels:"]
        lines.append("")
        lines.append("Standard Gmail folders (use short name in 'destination' / 'folder'):")
        for short, full in GMAIL_FOLDERS.items():
            lines.append(f"  {short:12} → {full}")

        lines.append("")
        lines.append("All folders (including custom labels):")
        for item in folder_list:
            if item is None:
                continue
            decoded = item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item)
            # Extract folder name from IMAP LIST response
            # Format: (\HasNoChildren) "/" "INBOX" or (\HasChildren) "/" "[Gmail]/Sent Mail"
            parts = decoded.split('"')
            if len(parts) >= 3:
                folder_name = parts[-2] if parts[-2] != "/" else parts[-1].strip()
            else:
                folder_name = decoded.split()[-1].strip('"')
            if folder_name:
                lines.append(f"  {folder_name}")

        return "\n".join(lines)
    except Exception as e:
        logger.warning("list_folders failed: %s", e)
        return f"Error listing folders: {e}"
    finally:
        try:
            mail.logout()
        except Exception:
            pass


# ── public async entrypoint ───────────────────────────────────────────────────

async def gmail_actions(
    action: str,
    # Send / reply / forward / draft
    to: str = "",
    subject: str = "",
    body: str = "",
    cc: str = "",
    bcc: str = "",
    # Reply / forward / delete / archive / move / mark_unread
    uid: str = "",
    # Reply-specific
    reply_all: bool = False,
    extra_to: str = "",
    # Forward-specific
    note: str = "",
    # Delete-specific
    permanent: bool = False,
    # Move-specific
    destination: str = "",
    # Folder context (which mailbox the UID lives in)
    folder: str = "inbox",
) -> str:
    """
    Unified async entrypoint for all Gmail write/management operations.

    Args:
        action:       which operation to perform (see actions below).
        to:           recipient address(es), comma-separated.
                      Accepts "Name <email>" or plain "email" format.
        subject:      email subject line. Required for send, draft.
        body:         plain text body. Required for send, reply, forward, draft.
        cc:           CC address(es), comma-separated. Optional.
        bcc:          BCC address(es), comma-separated. Optional for send only.
        uid:          IMAP UID of the email to act on. Get UIDs from gmail_check
                      (list_unread / list_recent / search output shows 'UID: <value>').
                      Required for reply, reply_all, forward, delete, archive,
                      move, mark_unread.
        reply_all:    if True, reply to all original recipients (not just sender).
                      Only used for reply action.
        extra_to:     additional recipients for reply, comma-separated. Optional.
        note:         text to prepend before the quoted original in a forward. Optional.
        permanent:    if True, delete_permanent skips Trash and expunges immediately.
                      UNRECOVERABLE. Default False.
        destination:  target folder for move action. Use short name (inbox/sent/drafts/
                      spam/trash/archive/starred) or full IMAP path. Required for move.
        folder:       source mailbox the UID lives in. Default 'inbox'.
                      Use when the email is not in INBOX (e.g. 'sent', 'spam').

    Actions:
        send            compose and send a new email.
        reply           reply to an email by UID.
        reply_all       reply to all recipients of an email by UID.
        forward         forward an email by UID to new recipients.
        delete          move email to Trash (recoverable).
        delete_permanent permanently delete email — no Trash, UNRECOVERABLE.
        archive         move email to All Mail (out of Inbox, not deleted).
        move            move email to a specific folder/label.
        mark_unread     mark a read email as unread.
        list_folders    list all folders and custom labels on this account.
        create_draft    save a draft to [Gmail]/Drafts.

    Returns a plain string in all cases. Never raises.
    """
    dispatch = {
        "send":             lambda: _send_email(to, subject, body, cc, bcc),
        "reply":            lambda: _reply_email(uid, body, reply_all=False, extra_to=extra_to),
        "reply_all":        lambda: _reply_email(uid, body, reply_all=True,  extra_to=extra_to),
        "forward":          lambda: _forward_email(uid, to, note),
        "delete":           lambda: _delete_email(uid, permanent=False, folder=folder),
        "delete_permanent": lambda: _delete_email(uid, permanent=True,  folder=folder),
        "archive":          lambda: _archive_email(uid, folder=folder),
        "move":             lambda: _move_email(uid, destination, source_folder=folder),
        "mark_unread":      lambda: _mark_unread(uid, folder=folder),
        "list_folders":     lambda: _list_folders(),
        "create_draft":     lambda: _create_draft(to, subject, body, cc),
    }

    fn = dispatch.get(action)
    if fn is None:
        return (
            f"Error: unknown action '{action}'. "
            f"Valid actions: {', '.join(dispatch.keys())}."
        )

    # Pre-call validation (catches missing required args before hitting the network)
    uid_required = {"reply", "reply_all", "forward", "delete", "delete_permanent",
                    "archive", "move", "mark_unread"}
    to_required  = {"send", "forward", "create_draft"}

    if action in uid_required and not uid.strip():
        return f"Error: '{action}' requires a 'uid'. Get UIDs from gmail_check list_unread/search output."

    if action in to_required and not to.strip():
        return f"Error: '{action}' requires a 'to' address."

    if action == "move" and not destination.strip():
        return "Error: 'move' requires a 'destination' folder name."

    if action in {"send", "create_draft"}:
        if not subject.strip():
            return f"Error: '{action}' requires a 'subject'."
        if not body.strip():
            return f"Error: '{action}' requires a 'body'."

    if action in {"reply", "reply_all", "forward"} and not body.strip() and action != "forward":
        return f"Error: '{action}' requires a 'body'."

    if action == "delete_permanent":
        # Surface the permanent-delete warning in the return if called without explicit confirmation
        # The LLM should confirm with the user before calling this
        logger.warning("delete_permanent called for UID %s — this is unrecoverable", uid)

    try:
        return await asyncio.to_thread(fn)
    except Exception as e:
        logger.exception("gmail_actions failed (action=%r): %s", action, e)
        return f"Unexpected error in gmail_actions '{action}': {e}"