"""
Gmail Check Tool — Reads unread inbox headers using standard IMAP protocol.
"""

import os
import imaplib
import email
from email.header import decode_header
import logging
import asyncio

logger = logging.getLogger(__name__)

def _fetch_unread_emails(max_emails: int) -> str:
    username = os.getenv("GMAIL_USER")
    password = os.getenv("GMAIL_APP_PASSWORD")

    if not username or not password:
        return "Error: GMAIL_USER or GMAIL_APP_PASSWORD environment variables are missing."

    try:
        # Connect to Gmail IMAP server
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(username, password)
        mail.select("inbox")

        # Search for all unread (UNSEEN) emails
        status, response = mail.search(None, "UNSEEN")
        mail_ids = response[0].split()

        if not mail_ids or mail_ids == [b'']:
            mail.logout()
            return "You have 0 unread emails in your inbox."

        # Grab the latest unread emails first
        mail_ids = mail_ids[::-1][:max_emails]
        output = [f"Recent Unread Emails (Total Unread Count: {len(response[0].split())}):"]

        for count, m_id in enumerate(mail_ids, 1):
            # Fetch the email headers
            status, data = mail.fetch(m_id, "(RFC822.HEADER)")
            raw_email = data[0][1]
            msg = email.message_from_bytes(raw_email)

            # Decode Subject
            subject, encoding = decode_header(msg["Subject"] or "No Subject")[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or "utf-8", errors="ignore")

            # Decode From address
            from_sender, encoding = decode_header(msg["From"] or "Unknown Sender")[0]
            if isinstance(from_sender, bytes):
                from_sender = from_sender.decode(encoding or "utf-8", errors="ignore")

            # Date string
            date_sent = msg["Date"] or "Unknown Date"

            output.append(f"{count}. FROM: {from_sender}\n   SUBJECT: {subject}\n   DATE: {date_sent}")

        mail.logout()
        return "\n\n".join(output)

    except Exception as e:
        logger.exception("Gmail IMAP reading error")
        return f"Failed to retrieve emails: {str(e)}"


async def check_gmail(action: str = "list_unread", max_results: int = 5) -> str:
    """
    Main async entrypoint for the email reading tool. 
    Runs the blocking IMAP operations inside a worker thread pool.
    """
    if action != "list_unread":
        return f"Error: Unknown action '{action}'."
        
    return await asyncio.to_thread(_fetch_unread_emails, max_results)