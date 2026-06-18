"""
Tests for runtime/tools/gmail_actions.py

Run from the project root:
    pytest test/test_gmail_actions.py -v

Structure:
    Unit tests      — no network. Mock SMTP and IMAP entirely.
                      These must always pass regardless of environment.
    Integration     — real SMTP/IMAP. Skipped unless env vars are set.
                      Run manually to verify against live Gmail account.

Coverage:
    Address validation  — valid/invalid/mixed, display-name format, empty
    Message building    — threading headers, CC/BCC, reply-all recipient logic
    send                — happy path, missing fields, bad addresses, SMTP failure
    reply               — threading headers set, reply-to-sender, IMAP fetch
    reply_all           — all recipients included, self excluded
    forward             — Fwd: prefix, body quoting, note prepended
    create_draft        — IMAP APPEND called with \\Draft flag
    delete              — MOVE tried first, COPY+DELETE fallback
    delete_permanent    — store \\Deleted + expunge called
    archive             — MOVE to All Mail, fallback
    move                — short name resolved to IMAP path, fallback
    mark_unread         — -FLAGS \\Seen called
    list_folders        — returns both standard and custom folders
    unknown action      — returns valid-actions list
    missing uid         — fires before any network call
    missing to          — fires before any network call
    missing credentials — returns error string, never raises
    IMAP failure        — returns error string, never raises
    SMTP failure        — returns error string, never raises
"""

import asyncio
import email as email_lib
import imaplib
import os
import smtplib
import sys
import time
import unittest
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch, call

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runtime"))

from runtime.tools.gmail_actions import (
    gmail_actions,
    _validate_address,
    _parse_address_list,
    _build_message,
    _get_credentials,
    GMAIL_FOLDERS,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


ENV_CREDS = {"GMAIL_USER": "me@gmail.com", "GMAIL_APP_PASSWORD": "testpass1234"}


def _make_raw_email(
    subject="Test Subject",
    from_="sender@example.com",
    to="me@gmail.com",
    cc="",
    message_id="<abc123@mail.gmail.com>",
    references="",
    body="Original email body.",
) -> bytes:
    """Build a minimal RFC 822 email as bytes for IMAP fetch mocking."""
    msg = MIMEMultipart("alternative")
    msg["Subject"]    = subject
    msg["From"]       = from_
    msg["To"]         = to
    if cc:
        msg["Cc"]     = cc
    msg["Message-ID"] = message_id
    if references:
        msg["References"] = references
    msg["Date"]       = "Thu, 18 Jun 2026 10:00:00 +0530"
    msg.attach(MIMEText(body, "plain", "utf-8"))
    return msg.as_bytes()


def _mock_smtp():
    """Return a MagicMock that behaves like smtplib.SMTP."""
    mock = MagicMock(spec=smtplib.SMTP)
    mock.sendmail.return_value = {}   # no rejected recipients
    return mock


def _mock_imap(fetch_raw: bytes | None = None, uid_responses: dict | None = None):
    """
    Return a MagicMock that behaves like imaplib.IMAP4_SSL.

    fetch_raw:      bytes returned for uid("fetch", ...) calls
    uid_responses:  dict mapping command → (status, data) for uid() calls
    """
    mock = MagicMock(spec=imaplib.IMAP4_SSL)
    mock.select.return_value = ("OK", [b"3"])
    mock.logout.return_value = ("BYE", [b""])
    mock.list.return_value   = (
        "OK",
        [b'(\\HasNoChildren) "/" "INBOX"',
         b'(\\HasNoChildren) "/" "[Gmail]/All Mail"',
         b'(\\HasNoChildren) "/" "[Gmail]/Drafts"',
         b'(\\HasNoChildren) "/" "MyLabel"'],
    )
    mock.expunge.return_value = ("OK", [None])
    mock.append.return_value  = ("OK", [b"[APPENDUID 123 456]"])

    def uid_side_effect(command, *args):
        if uid_responses and command in uid_responses:
            return uid_responses[command]
        if command == "fetch":
            if fetch_raw:
                return ("OK", [(None, fetch_raw)])
            return ("NO", [None])
        if command in ("store", "copy", "MOVE"):
            return ("OK", [b""])
        if command == "search":
            return ("OK", [b"101"])
        return ("OK", [b""])

    mock.uid.side_effect = uid_side_effect
    return mock


# ── address validation tests ──────────────────────────────────────────────────

class TestValidateAddress(unittest.TestCase):

    def test_plain_email_valid(self):
        self.assertEqual(_validate_address("user@example.com"), "user@example.com")

    def test_display_name_format(self):
        self.assertEqual(_validate_address("Alice <alice@example.com>"), "alice@example.com")

    def test_invalid_no_at(self):
        self.assertIsNone(_validate_address("notanemail"))

    def test_invalid_double_at(self):
        self.assertIsNone(_validate_address("bad@@domain.com"))

    def test_empty_string(self):
        self.assertIsNone(_validate_address(""))

    def test_plus_addressing(self):
        self.assertEqual(_validate_address("user+tag@domain.co.uk"), "user+tag@domain.co.uk")

    def test_whitespace_stripped(self):
        self.assertEqual(_validate_address("  user@example.com  "), "user@example.com")


class TestParseAddressList(unittest.TestCase):

    def test_single_valid(self):
        valid, invalid = _parse_address_list("alice@example.com")
        self.assertEqual(valid, ["alice@example.com"])
        self.assertEqual(invalid, [])

    def test_multiple_valid(self):
        valid, invalid = _parse_address_list("a@x.com, b@y.com, c@z.com")
        self.assertEqual(len(valid), 3)
        self.assertEqual(invalid, [])

    def test_mixed_valid_invalid(self):
        valid, invalid = _parse_address_list("good@x.com, notanemail, bad@@y.com")
        self.assertEqual(valid, ["good@x.com"])
        self.assertEqual(len(invalid), 2)

    def test_display_name_format(self):
        valid, invalid = _parse_address_list("Alice <alice@x.com>, Bob <bob@y.com>")
        self.assertIn("alice@x.com", valid)
        self.assertIn("bob@y.com", valid)
        self.assertEqual(invalid, [])

    def test_empty_string(self):
        valid, invalid = _parse_address_list("")
        self.assertEqual(valid, [])
        self.assertEqual(invalid, [])

    def test_none_equivalent(self):
        valid, invalid = _parse_address_list("   ")
        self.assertEqual(valid, [])


# ── message building tests ────────────────────────────────────────────────────

class TestBuildMessage(unittest.TestCase):

    def test_basic_fields_set(self):
        msg = _build_message("me@g.com", ["alice@x.com"], "Hello", "Body text")
        self.assertEqual(msg["To"], "alice@x.com")
        self.assertEqual(msg["Subject"], "Hello")
        self.assertIn("me@g.com", msg["From"])
        self.assertIsNotNone(msg["Message-ID"])

    def test_cc_set(self):
        msg = _build_message("me@g.com", ["a@x.com"], "S", "B", cc=["b@y.com"])
        self.assertIn("b@y.com", msg["Cc"])

    def test_bcc_not_in_headers(self):
        """BCC addresses must NOT appear in the message headers."""
        msg = _build_message("me@g.com", ["a@x.com"], "S", "B", bcc=["secret@x.com"])
        self.assertIsNone(msg["Bcc"])

    def test_reply_threading_headers(self):
        msg = _build_message(
            "me@g.com", ["a@x.com"], "Re: Test", "Reply",
            reply_to_message_id="<orig@mail>",
            references="<prev@mail>",
        )
        self.assertEqual(msg["In-Reply-To"], "<orig@mail>")
        self.assertIn("<orig@mail>", msg["References"])
        self.assertIn("<prev@mail>", msg["References"])

    def test_no_reply_no_threading_headers(self):
        msg = _build_message("me@g.com", ["a@x.com"], "New", "Body")
        self.assertIsNone(msg.get("In-Reply-To"))
        self.assertIsNone(msg.get("References"))

    def test_multiple_to_addresses(self):
        msg = _build_message("me@g.com", ["a@x.com", "b@y.com"], "S", "B")
        self.assertIn("a@x.com", msg["To"])
        self.assertIn("b@y.com", msg["To"])


# ── dispatch and validation tests ─────────────────────────────────────────────

class TestDispatchValidation(unittest.TestCase):
    """These tests check that validation fires BEFORE any network call."""

    def test_unknown_action(self):
        result = run(gmail_actions(action="teleport"))
        self.assertIn("unknown action", result.lower())
        self.assertIn("teleport", result)

    def test_send_missing_to(self):
        result = run(gmail_actions(action="send", to="", subject="S", body="B"))
        self.assertIn("Error", result)
        self.assertIn("to", result)

    def test_send_missing_subject(self):
        result = run(gmail_actions(action="send", to="a@x.com", subject="", body="B"))
        self.assertIn("Error", result)
        self.assertIn("subject", result)

    def test_send_missing_body(self):
        result = run(gmail_actions(action="send", to="a@x.com", subject="S", body=""))
        self.assertIn("Error", result)
        self.assertIn("body", result)

    def test_reply_missing_uid(self):
        result = run(gmail_actions(action="reply", uid="", body="My reply"))
        self.assertIn("Error", result)
        self.assertIn("uid", result)

    def test_reply_missing_body(self):
        result = run(gmail_actions(action="reply", uid="101", body=""))
        self.assertIn("Error", result)
        self.assertIn("body", result)

    def test_reply_all_missing_uid(self):
        result = run(gmail_actions(action="reply_all", uid="", body="Reply"))
        self.assertIn("Error", result)

    def test_forward_missing_uid(self):
        result = run(gmail_actions(action="forward", uid="", to="a@x.com"))
        self.assertIn("Error", result)
        self.assertIn("uid", result)

    def test_forward_missing_to(self):
        result = run(gmail_actions(action="forward", uid="101", to=""))
        self.assertIn("Error", result)
        self.assertIn("to", result)

    def test_delete_missing_uid(self):
        result = run(gmail_actions(action="delete", uid=""))
        self.assertIn("Error", result)

    def test_archive_missing_uid(self):
        result = run(gmail_actions(action="archive", uid=""))
        self.assertIn("Error", result)

    def test_move_missing_uid(self):
        result = run(gmail_actions(action="move", uid="", destination="trash"))
        self.assertIn("Error", result)

    def test_move_missing_destination(self):
        result = run(gmail_actions(action="move", uid="101", destination=""))
        self.assertIn("Error", result)
        self.assertIn("destination", result)

    def test_mark_unread_missing_uid(self):
        result = run(gmail_actions(action="mark_unread", uid=""))
        self.assertIn("Error", result)

    def test_create_draft_missing_to(self):
        result = run(gmail_actions(action="create_draft", to="", subject="S", body="B"))
        self.assertIn("Error", result)

    def test_send_invalid_address(self):
        result = run(gmail_actions(action="send", to="notanemail", subject="S", body="B"))
        self.assertIn("Error", result)
        self.assertIn("valid", result.lower())


# ── missing credentials ───────────────────────────────────────────────────────

class TestMissingCredentials(unittest.TestCase):
    """All actions must return an error string when env vars are not set."""

    def _no_creds(self):
        env = dict(os.environ)
        env.pop("GMAIL_USER", None)
        env.pop("GMAIL_APP_PASSWORD", None)
        return env

    def _run_no_creds(self, **kwargs):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GMAIL_USER", None)
            os.environ.pop("GMAIL_APP_PASSWORD", None)
            return run(gmail_actions(**kwargs))

    def test_send(self):
        r = self._run_no_creds(action="send", to="a@x.com", subject="S", body="B")
        self.assertIn("Error", r)

    def test_reply(self):
        r = self._run_no_creds(action="reply", uid="101", body="B")
        self.assertIn("Error", r)

    def test_delete(self):
        r = self._run_no_creds(action="delete", uid="101")
        self.assertIn("Error", r)

    def test_archive(self):
        r = self._run_no_creds(action="archive", uid="101")
        self.assertIn("Error", r)

    def test_list_folders(self):
        r = self._run_no_creds(action="list_folders")
        self.assertIn("Error", r)


# ── send tests ────────────────────────────────────────────────────────────────

class TestSend(unittest.TestCase):

    @patch("tools.gmail_actions._smtp_connect")
    def test_send_happy_path(self, mock_connect):
        mock_smtp = _mock_smtp()
        mock_connect.return_value = mock_smtp
        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(
                action="send",
                to="alice@example.com",
                subject="Hello",
                body="Test body",
            ))
        self.assertIn("Sent", result)
        self.assertIn("Hello", result)
        self.assertIn("alice@example.com", result)
        mock_smtp.sendmail.assert_called_once()
        mock_smtp.quit.assert_called_once()

    @patch("tools.gmail_actions._smtp_connect")
    def test_send_with_cc_and_bcc(self, mock_connect):
        mock_smtp = _mock_smtp()
        mock_connect.return_value = mock_smtp
        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(
                action="send",
                to="alice@example.com",
                subject="Hello",
                body="Body",
                cc="carol@example.com",
                bcc="secret@example.com",
            ))
        self.assertIn("Sent", result)
        self.assertIn("CC", result)
        self.assertIn("BCC", result)
        # Verify sendmail got all 3 recipients
        call_args = mock_smtp.sendmail.call_args
        recipients = call_args[0][1]
        self.assertIn("alice@example.com", recipients)
        self.assertIn("carol@example.com", recipients)
        self.assertIn("secret@example.com", recipients)

    @patch("tools.gmail_actions._smtp_connect")
    def test_send_bcc_not_in_headers(self, mock_connect):
        """BCC must never appear in the message body sent to SMTP."""
        mock_smtp = _mock_smtp()
        mock_connect.return_value = mock_smtp
        with patch.dict(os.environ, ENV_CREDS):
            run(gmail_actions(
                action="send",
                to="alice@example.com",
                subject="S",
                body="B",
                bcc="secret@example.com",
            ))
        # The message string passed to sendmail must not contain secret@example.com in headers
        msg_str = mock_smtp.sendmail.call_args[0][2]
        # BCC address should only appear in the SMTP envelope, not headers
        header_section = msg_str.split("\n\n")[0]
        self.assertNotIn("secret@example.com", header_section)

    @patch("tools.gmail_actions._smtp_connect")
    def test_send_multiple_to(self, mock_connect):
        mock_smtp = _mock_smtp()
        mock_connect.return_value = mock_smtp
        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(
                action="send",
                to="alice@example.com, bob@example.com",
                subject="S",
                body="B",
            ))
        self.assertIn("Sent", result)
        recipients = mock_smtp.sendmail.call_args[0][1]
        self.assertIn("alice@example.com", recipients)
        self.assertIn("bob@example.com", recipients)

    @patch("tools.gmail_actions._smtp_connect")
    def test_send_smtp_connection_failure(self, mock_connect):
        mock_connect.return_value = None
        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(action="send", to="a@x.com", subject="S", body="B"))
        self.assertIn("Error", result)
        self.assertIn("SMTP", result)

    @patch("tools.gmail_actions._smtp_connect")
    def test_send_smtp_exception(self, mock_connect):
        mock_smtp = _mock_smtp()
        mock_smtp.sendmail.side_effect = smtplib.SMTPException("Connection lost")
        mock_connect.return_value = mock_smtp
        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(action="send", to="a@x.com", subject="S", body="B"))
        self.assertIn("Error", result)
        self.assertIsInstance(result, str)

    @patch("tools.gmail_actions._smtp_connect")
    def test_send_quit_called_even_on_error(self, mock_connect):
        """SMTP connection must always be closed even when sendmail raises."""
        mock_smtp = _mock_smtp()
        mock_smtp.sendmail.side_effect = smtplib.SMTPException("Oops")
        mock_connect.return_value = mock_smtp
        with patch.dict(os.environ, ENV_CREDS):
            run(gmail_actions(action="send", to="a@x.com", subject="S", body="B"))
        mock_smtp.quit.assert_called_once()


# ── reply tests ───────────────────────────────────────────────────────────────

class TestReply(unittest.TestCase):

    @patch("tools.gmail_actions._smtp_connect")
    @patch("tools.gmail_actions._imap_connect")
    def test_reply_sets_threading_headers(self, mock_imap_conn, mock_smtp_conn):
        raw = _make_raw_email(
            subject="Original subject",
            from_="sender@example.com",
            message_id="<orig123@mail>",
            references="<prev@mail>",
        )
        mock_imap_conn.return_value = _mock_imap(fetch_raw=raw)
        mock_smtp = _mock_smtp()
        mock_smtp_conn.return_value = mock_smtp

        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(action="reply", uid="101", body="My reply"))

        self.assertIn("Reply sent", result)
        # Check the message that was actually sent
        msg_str = mock_smtp.sendmail.call_args[0][2]
        self.assertIn("In-Reply-To: <orig123@mail>", msg_str)
        self.assertIn("<orig123@mail>", msg_str)   # In References
        self.assertIn("Re: Original subject", msg_str)

    @patch("tools.gmail_actions._smtp_connect")
    @patch("tools.gmail_actions._imap_connect")
    def test_reply_does_not_send_to_self(self, mock_imap_conn, mock_smtp_conn):
        """Reply must not include our own address in the recipient list."""
        raw = _make_raw_email(
            from_="me@gmail.com",   # sender IS us — should not reply to self
            to="me@gmail.com",
        )
        mock_imap_conn.return_value = _mock_imap(fetch_raw=raw)
        mock_smtp = _mock_smtp()
        mock_smtp_conn.return_value = mock_smtp

        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(action="reply", uid="101", body="Body"))

        # When original sender == us, there are no valid recipients
        self.assertIn("Error", result)
        self.assertIn("recipient", result.lower())

    @patch("tools.gmail_actions._smtp_connect")
    @patch("tools.gmail_actions._imap_connect")
    def test_reply_all_includes_cc(self, mock_imap_conn, mock_smtp_conn):
        raw = _make_raw_email(
            from_="sender@example.com",
            to="me@gmail.com",
            cc="carol@example.com",
        )
        mock_imap_conn.return_value = _mock_imap(fetch_raw=raw)
        mock_smtp = _mock_smtp()
        mock_smtp_conn.return_value = mock_smtp

        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(action="reply_all", uid="101", body="All reply"))

        self.assertIn("Reply-all sent", result)
        recipients = mock_smtp.sendmail.call_args[0][1]
        self.assertIn("sender@example.com", recipients)
        self.assertIn("carol@example.com", recipients)

    @patch("tools.gmail_actions._smtp_connect")
    @patch("tools.gmail_actions._imap_connect")
    def test_reply_all_excludes_self(self, mock_imap_conn, mock_smtp_conn):
        """Our own address must never appear in reply-all recipients."""
        raw = _make_raw_email(
            from_="sender@example.com",
            to="me@gmail.com, other@example.com",
            cc="me@gmail.com",
        )
        mock_imap_conn.return_value = _mock_imap(fetch_raw=raw)
        mock_smtp = _mock_smtp()
        mock_smtp_conn.return_value = mock_smtp

        with patch.dict(os.environ, ENV_CREDS):
            run(gmail_actions(action="reply_all", uid="101", body="Body"))

        recipients = mock_smtp.sendmail.call_args[0][1]
        self.assertNotIn("me@gmail.com", recipients)

    @patch("tools.gmail_actions._imap_connect")
    def test_reply_imap_fetch_failure(self, mock_imap_conn):
        """When IMAP fetch fails, return a clear error, never crash."""
        mock_imap_conn.return_value = _mock_imap(fetch_raw=None)
        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(action="reply", uid="999", body="Body"))
        self.assertIn("Error", result)
        self.assertIsInstance(result, str)


# ── forward tests ─────────────────────────────────────────────────────────────

class TestForward(unittest.TestCase):

    @patch("tools.gmail_actions._smtp_connect")
    @patch("tools.gmail_actions._imap_connect")
    def test_forward_adds_fwd_prefix(self, mock_imap_conn, mock_smtp_conn):
        raw = _make_raw_email(subject="Original")
        mock_imap_conn.return_value = _mock_imap(fetch_raw=raw)
        mock_smtp = _mock_smtp()
        mock_smtp_conn.return_value = mock_smtp

        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(action="forward", uid="101", to="fwd@example.com"))

        self.assertIn("Forwarded", result)
        msg_str = mock_smtp.sendmail.call_args[0][2]
        self.assertIn("Fwd: Original", msg_str)

    @patch("tools.gmail_actions._smtp_connect")
    @patch("tools.gmail_actions._imap_connect")
    def test_forward_does_not_double_prefix(self, mock_imap_conn, mock_smtp_conn):
        """If subject already starts with Fwd:, must not add another one."""
        raw = _make_raw_email(subject="Fwd: Already forwarded")
        mock_imap_conn.return_value = _mock_imap(fetch_raw=raw)
        mock_smtp = _mock_smtp()
        mock_smtp_conn.return_value = mock_smtp

        with patch.dict(os.environ, ENV_CREDS):
            run(gmail_actions(action="forward", uid="101", to="a@x.com"))

        msg_str = mock_smtp.sendmail.call_args[0][2]
        self.assertNotIn("Fwd: Fwd:", msg_str)

    @patch("tools.gmail_actions._smtp_connect")
    @patch("tools.gmail_actions._imap_connect")
    def test_forward_note_prepended(self, mock_imap_conn, mock_smtp_conn):
        raw = _make_raw_email(body="Original body")
        mock_imap_conn.return_value = _mock_imap(fetch_raw=raw)
        mock_smtp = _mock_smtp()
        mock_smtp_conn.return_value = mock_smtp

        with patch.dict(os.environ, ENV_CREDS):
            run(gmail_actions(action="forward", uid="101", to="a@x.com", note="FYI check this out"))

        msg_str = mock_smtp.sendmail.call_args[0][2]
        self.assertIn("FYI check this out", msg_str)
        self.assertIn("Original body", msg_str)

    @patch("tools.gmail_actions._smtp_connect")
    @patch("tools.gmail_actions._imap_connect")
    def test_forward_original_body_quoted(self, mock_imap_conn, mock_smtp_conn):
        raw = _make_raw_email(body="The original content here")
        mock_imap_conn.return_value = _mock_imap(fetch_raw=raw)
        mock_smtp = _mock_smtp()
        mock_smtp_conn.return_value = mock_smtp

        with patch.dict(os.environ, ENV_CREDS):
            run(gmail_actions(action="forward", uid="101", to="a@x.com"))

        msg_str = mock_smtp.sendmail.call_args[0][2]
        self.assertIn("The original content here", msg_str)
        self.assertIn("Forwarded message", msg_str)


# ── delete tests ──────────────────────────────────────────────────────────────

class TestDelete(unittest.TestCase):

    @patch("tools.gmail_actions._imap_connect")
    def test_delete_tries_move_first(self, mock_imap_conn):
        """delete should try MOVE extension before falling back to COPY+DELETE."""
        mock_imap = _mock_imap()
        mock_imap_conn.return_value = mock_imap

        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(action="delete", uid="101"))

        self.assertIn("Trash", result)
        # MOVE should be attempted
        uid_calls = [str(c) for c in mock_imap.uid.call_args_list]
        self.assertTrue(any("MOVE" in c for c in uid_calls))

    @patch("tools.gmail_actions._imap_connect")
    def test_delete_copy_fallback_when_move_fails(self, mock_imap_conn):
        """When MOVE raises IMAP4.error, fall back to copy + store + expunge."""
        mock_imap = _mock_imap()

        def uid_side(*args):
            if args[0] == "MOVE":
                raise imaplib.IMAP4.error("MOVE not supported")
            return ("OK", [b""])

        mock_imap.uid.side_effect = uid_side
        mock_imap_conn.return_value = mock_imap

        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(action="delete", uid="101"))

        self.assertIn("Trash", result)
        mock_imap.expunge.assert_called()

    @patch("tools.gmail_actions._imap_connect")
    def test_delete_permanent_sets_deleted_flag_and_expunges(self, mock_imap_conn):
        mock_imap = _mock_imap()
        mock_imap_conn.return_value = mock_imap

        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(action="delete_permanent", uid="101"))

        self.assertIn("Permanently deleted", result)
        self.assertIn("unrecoverable", result.lower())

        uid_calls = [str(c) for c in mock_imap.uid.call_args_list]
        self.assertTrue(any("store" in c.lower() and "Deleted" in c for c in uid_calls))
        mock_imap.expunge.assert_called_once()

    @patch("tools.gmail_actions._imap_connect")
    def test_delete_logout_always_called(self, mock_imap_conn):
        mock_imap = _mock_imap()
        mock_imap_conn.return_value = mock_imap

        with patch.dict(os.environ, ENV_CREDS):
            run(gmail_actions(action="delete", uid="101"))

        mock_imap.logout.assert_called()


# ── archive tests ─────────────────────────────────────────────────────────────

class TestArchive(unittest.TestCase):

    @patch("tools.gmail_actions._imap_connect")
    def test_archive_moves_to_all_mail(self, mock_imap_conn):
        mock_imap = _mock_imap()
        mock_imap_conn.return_value = mock_imap

        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(action="archive", uid="101"))

        self.assertIn("Archived", result)
        self.assertIn("All Mail", result)

    @patch("tools.gmail_actions._imap_connect")
    def test_archive_fallback_on_move_error(self, mock_imap_conn):
        mock_imap = _mock_imap()

        def uid_side(*args):
            if args[0] == "MOVE":
                raise imaplib.IMAP4.error("Not supported")
            return ("OK", [b""])

        mock_imap.uid.side_effect = uid_side
        mock_imap_conn.return_value = mock_imap

        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(action="archive", uid="101"))

        self.assertIn("Archived", result)
        mock_imap.expunge.assert_called()


# ── move tests ────────────────────────────────────────────────────────────────

class TestMove(unittest.TestCase):

    @patch("tools.gmail_actions._imap_connect")
    def test_move_short_name_resolved(self, mock_imap_conn):
        """Short folder names like 'spam' must be resolved to full IMAP paths."""
        mock_imap = _mock_imap()
        mock_imap_conn.return_value = mock_imap

        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(action="move", uid="101", destination="spam"))

        self.assertIn("[Gmail]/Spam", result)

    @patch("tools.gmail_actions._imap_connect")
    def test_move_custom_label(self, mock_imap_conn):
        """Custom labels are passed through as-is."""
        mock_imap = _mock_imap()
        mock_imap_conn.return_value = mock_imap

        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(action="move", uid="101", destination="MyLabel"))

        self.assertIn("MyLabel", result)

    @patch("tools.gmail_actions._imap_connect")
    def test_move_uses_source_folder(self, mock_imap_conn):
        """Source folder should be selected before moving."""
        mock_imap = _mock_imap()
        mock_imap_conn.return_value = mock_imap

        with patch.dict(os.environ, ENV_CREDS):
            run(gmail_actions(action="move", uid="101", destination="trash", folder="spam"))

        select_calls = [str(c) for c in mock_imap.select.call_args_list]
        self.assertTrue(any("Spam" in c for c in select_calls))


# ── mark_unread tests ─────────────────────────────────────────────────────────

class TestMarkUnread(unittest.TestCase):

    @patch("tools.gmail_actions._imap_connect")
    def test_mark_unread_removes_seen_flag(self, mock_imap_conn):
        mock_imap = _mock_imap()
        mock_imap_conn.return_value = mock_imap

        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(action="mark_unread", uid="101"))

        self.assertIn("unread", result.lower())
        uid_calls = [str(c) for c in mock_imap.uid.call_args_list]
        # -FLAGS and \Seen must appear in the call
        self.assertTrue(
            any("-FLAGS" in c and "Seen" in c for c in uid_calls),
            f"Expected -FLAGS \\Seen call, got: {uid_calls}"
        )


# ── create_draft tests ────────────────────────────────────────────────────────

class TestCreateDraft(unittest.TestCase):

    @patch("tools.gmail_actions._imap_connect")
    def test_create_draft_uses_append(self, mock_imap_conn):
        mock_imap = _mock_imap()
        mock_imap_conn.return_value = mock_imap

        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(
                action="create_draft",
                to="alice@example.com",
                subject="Draft subject",
                body="Draft body",
            ))

        self.assertIn("Draft saved", result)
        self.assertIn("Draft subject", result)
        mock_imap.append.assert_called_once()

        # Verify it was appended to the Drafts folder
        append_args = mock_imap.append.call_args[0]
        self.assertIn("Drafts", append_args[0])

    @patch("tools.gmail_actions._imap_connect")
    def test_create_draft_draft_flag_set(self, mock_imap_conn):
        mock_imap = _mock_imap()
        mock_imap_conn.return_value = mock_imap

        with patch.dict(os.environ, ENV_CREDS):
            run(gmail_actions(
                action="create_draft",
                to="a@x.com",
                subject="S",
                body="B",
            ))

        append_args = mock_imap.append.call_args[0]
        self.assertIn("\\Draft", append_args[1])

    @patch("tools.gmail_actions._imap_connect")
    def test_create_draft_imap_failure_returns_error(self, mock_imap_conn):
        mock_imap = _mock_imap()
        mock_imap.append.return_value = ("NO", [b"Permission denied"])
        mock_imap_conn.return_value = mock_imap

        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(
                action="create_draft",
                to="a@x.com",
                subject="S",
                body="B",
            ))

        self.assertIn("Error", result)


# ── list_folders tests ────────────────────────────────────────────────────────

class TestListFolders(unittest.TestCase):

    @patch("tools.gmail_actions._imap_connect")
    def test_list_folders_includes_standard_names(self, mock_imap_conn):
        mock_imap_conn.return_value = _mock_imap()
        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(action="list_folders"))

        for short_name in ("inbox", "sent", "trash", "archive", "spam", "drafts"):
            self.assertIn(short_name, result)

    @patch("tools.gmail_actions._imap_connect")
    def test_list_folders_includes_custom_label(self, mock_imap_conn):
        mock_imap_conn.return_value = _mock_imap()
        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(action="list_folders"))

        self.assertIn("MyLabel", result)

    @patch("tools.gmail_actions._imap_connect")
    def test_list_folders_imap_failure(self, mock_imap_conn):
        mock_imap_conn.return_value = None
        with patch.dict(os.environ, ENV_CREDS):
            result = run(gmail_actions(action="list_folders"))
        self.assertIn("Error", result)


# ── IMAP connection failure (all IMAP actions) ────────────────────────────────

class TestImapConnectionFailure(unittest.TestCase):

    @patch("tools.gmail_actions._imap_connect")
    def test_delete_imap_none(self, mock_conn):
        mock_conn.return_value = None
        with patch.dict(os.environ, ENV_CREDS):
            r = run(gmail_actions(action="delete", uid="101"))
        self.assertIn("Error", r)

    @patch("tools.gmail_actions._imap_connect")
    def test_archive_imap_none(self, mock_conn):
        mock_conn.return_value = None
        with patch.dict(os.environ, ENV_CREDS):
            r = run(gmail_actions(action="archive", uid="101"))
        self.assertIn("Error", r)

    @patch("tools.gmail_actions._imap_connect")
    def test_move_imap_none(self, mock_conn):
        mock_conn.return_value = None
        with patch.dict(os.environ, ENV_CREDS):
            r = run(gmail_actions(action="move", uid="101", destination="trash"))
        self.assertIn("Error", r)

    @patch("tools.gmail_actions._imap_connect")
    def test_mark_unread_imap_none(self, mock_conn):
        mock_conn.return_value = None
        with patch.dict(os.environ, ENV_CREDS):
            r = run(gmail_actions(action="mark_unread", uid="101"))
        self.assertIn("Error", r)


# ── folder name resolution ────────────────────────────────────────────────────

class TestFolderNameResolution(unittest.TestCase):
    """Verify short names map to correct Gmail IMAP paths."""

    def test_all_short_names_map(self):
        expected = {
            "inbox":     "INBOX",
            "sent":      "[Gmail]/Sent Mail",
            "drafts":    "[Gmail]/Drafts",
            "spam":      "[Gmail]/Spam",
            "trash":     "[Gmail]/Trash",
            "archive":   "[Gmail]/All Mail",
            "starred":   "[Gmail]/Starred",
            "important": "[Gmail]/Important",
        }
        for short, full in expected.items():
            self.assertEqual(
                GMAIL_FOLDERS[short], full,
                f"Folder mapping wrong for '{short}': expected '{full}', got '{GMAIL_FOLDERS[short]}'"
            )

    def test_unknown_short_name_passed_through(self):
        """Names not in GMAIL_FOLDERS should be passed through as-is."""
        result = GMAIL_FOLDERS.get("nonexistent_label", "nonexistent_label")
        self.assertEqual(result, "nonexistent_label")


# ── integration tests (skipped without real credentials) ─────────────────────

HAVE_CREDS = bool(os.getenv("GMAIL_USER") and os.getenv("GMAIL_APP_PASSWORD"))

@unittest.skipUnless(HAVE_CREDS, "GMAIL_USER and GMAIL_APP_PASSWORD not set — skipping integration tests")
class TestGmailActionsIntegration(unittest.TestCase):
    """
    Live tests against a real Gmail account.

    Run manually:
        GMAIL_USER=you@gmail.com GMAIL_APP_PASSWORD=xxxx \
        pytest test/test_gmail_actions.py -v -k integration

    WARNING: test_send will send a real email to yourself.
             test_delete and test_archive act on real emails.
             Use a test/throwaway Gmail account.
    """

    def test_list_folders_live(self):
        result = run(gmail_actions(action="list_folders"))
        self.assertIsInstance(result, str)
        self.assertNotIn("Error", result)
        self.assertIn("INBOX", result)

    def test_send_self(self):
        """Send an email to yourself — verifiable in Gmail."""
        me = os.getenv("GMAIL_USER")
        result = run(gmail_actions(
            action="send",
            to=me,
            subject=f"Kairos integration test {int(time.time())}",
            body="This is an automated test from Kairos gmail_actions integration tests.",
        ))
        self.assertIn("Sent", result)
        self.assertNotIn("Error", result)

    def test_create_draft_live(self):
        me = os.getenv("GMAIL_USER")
        result = run(gmail_actions(
            action="create_draft",
            to=me,
            subject=f"Kairos draft test {int(time.time())}",
            body="Draft created by integration test.",
        ))
        self.assertIn("Draft saved", result)
        self.assertNotIn("Error", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)