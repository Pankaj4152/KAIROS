"""
Tests for runtime/tools/gmail_check.py

Run from the project root:
    pytest test/test_gmail_check.py -v

Structure:
    Unit tests   — no network, no credentials. Mock imaplib entirely.
                   These must always pass regardless of environment.
    Integration  — real IMAP connection to Gmail. Skipped automatically
                   when GMAIL_USER / GMAIL_APP_PASSWORD are not set.
                   Run these manually to verify the live tool.

What is tested:
    - Every action dispatches correctly
    - Header decoding handles RFC 2047 encoded strings (non-ASCII)
    - HTML body stripping produces clean plain text
    - Connection failures return error strings, never raise
    - Missing env vars caught before any network call
    - UID-based operations (get_body, mark_read) validate the uid arg
    - max_results clamping (1–20)
    - Unknown action returns the valid-actions list
    - Required-arg validation fires before hitting the network
"""

import asyncio
import email
import imaplib
import os
import sys
import unittest
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch, PropertyMock

# ── path setup ────────────────────────────────────────────────────────────────
# Adjust if your project layout differs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runtime"))

from runtime.tools.gmail_check import (
    check_gmail,
    _decode_header_value,
    _strip_html,
    _extract_body,
    _get_credentials,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def run(coro):
    """Run an async coroutine in tests without pytest-asyncio."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_simple_email(
    subject: str = "Test Subject",
    sender: str = "sender@example.com",
    date: str = "Thu, 18 Jun 2026 10:00:00 +0530",
    body: str = "Hello from test.",
    content_type: str = "plain",
) -> bytes:
    """Build a minimal RFC 822 email as bytes."""
    msg = MIMEText(body, content_type, "utf-8")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["Date"]    = date
    return msg.as_bytes()


def _make_multipart_email(plain: str = "", html: str = "") -> bytes:
    """Build a multipart/alternative email with both plain and HTML parts."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Multipart"
    msg["From"]    = "test@example.com"
    msg["Date"]    = "Thu, 18 Jun 2026 10:00:00 +0530"
    if plain:
        msg.attach(MIMEText(plain, "plain", "utf-8"))
    if html:
        msg.attach(MIMEText(html, "html", "utf-8"))
    return msg.as_bytes()


def _mock_imap(uid_list: list[bytes], fetch_data: dict[bytes, bytes]):
    """
    Build a mock imaplib.IMAP4_SSL instance.

    uid_list:    bytes returned from uid("search", ...) — e.g. [b"101", b"102"]
    fetch_data:  maps uid bytes → raw email bytes for uid("fetch", ...)
    """
    mock = MagicMock(spec=imaplib.IMAP4_SSL)
    mock.__enter__ = lambda s: s
    mock.__exit__  = MagicMock(return_value=False)

    # uid("search", ...) → ("OK", [b"101 102"])
    # uid("fetch", uid, "(RFC822.HEADER)") → ("OK", [(None, raw_bytes)])
    def uid_dispatch(command, *args):
        if command == "search":
            uid_bytes = b" ".join(uid_list) if uid_list else b""
            return ("OK", [uid_bytes])
        if command == "fetch":
            target_uid = args[0]
            raw = fetch_data.get(target_uid, None)
            if raw is None:
                return ("NO", [None])
            return ("OK", [(None, raw)])
        if command == "store":
            return ("OK", [b""])
        return ("OK", [b""])

    mock.uid.side_effect = uid_dispatch
    mock.select.return_value = ("OK", [b"3"])
    mock.logout.return_value = ("BYE", [b""])
    return mock


# ── unit tests ────────────────────────────────────────────────────────────────

class TestDecodeHeaderValue(unittest.TestCase):
    """_decode_header_value handles plain ASCII, encoded UTF-8, and None."""

    def test_plain_ascii(self):
        self.assertEqual(_decode_header_value("Hello World"), "Hello World")

    def test_none_returns_empty(self):
        self.assertEqual(_decode_header_value(None), "")

    def test_empty_string(self):
        self.assertEqual(_decode_header_value(""), "")

    def test_rfc2047_utf8(self):
        # =?utf-8?b?...?= — base64 encoded "नमस्ते"
        import base64
        encoded = base64.b64encode("नमस्ते".encode("utf-8")).decode()
        header  = f"=?utf-8?b?{encoded}?="
        result  = _decode_header_value(header)
        self.assertIn("नमस्ते", result)

    def test_rfc2047_quoted_printable(self):
        # Simple latin encoded header
        header = "=?iso-8859-1?q?Caf=E9?="
        result = _decode_header_value(header)
        self.assertIn("Caf", result)   # at minimum, no crash

    def test_mixed_encoded_and_plain(self):
        # Some headers mix encoded and plain segments
        header = "Hello =?utf-8?b?V29ybGQ=?="   # "Hello World"
        result = _decode_header_value(header)
        self.assertIn("Hello", result)
        self.assertIn("World", result)


class TestStripHtml(unittest.TestCase):
    """_strip_html extracts plain text from HTML."""

    def test_simple_paragraph(self):
        result = _strip_html("<p>Hello World</p>")
        self.assertIn("Hello World", result)

    def test_strips_tags_completely(self):
        result = _strip_html("<div><b>Bold</b> and <i>italic</i></div>")
        self.assertNotIn("<", result)
        self.assertIn("Bold", result)
        self.assertIn("italic", result)

    def test_empty_html(self):
        self.assertEqual(_strip_html(""), "")

    def test_nested_tags(self):
        html = "<html><body><h1>Title</h1><p>Paragraph text here.</p></body></html>"
        result = _strip_html(html)
        self.assertIn("Title", result)
        self.assertIn("Paragraph text here", result)
        self.assertNotIn("<", result)

    def test_whitespace_collapsed(self):
        html = "<p>Line   one</p><p>Line   two</p>"
        result = _strip_html(html)
        # Should not have excessive whitespace
        self.assertNotIn("   ", result)


class TestExtractBody(unittest.TestCase):
    """_extract_body picks plain text over HTML and truncates correctly."""

    def test_plain_text_email(self):
        raw = _make_simple_email(body="Plain body text here.")
        msg = email.message_from_bytes(raw)
        result = _extract_body(msg)
        self.assertIn("Plain body text here.", result)

    def test_html_only_email(self):
        raw = _make_multipart_email(html="<p>HTML only content</p>")
        msg = email.message_from_bytes(raw)
        result = _extract_body(msg)
        self.assertIn("HTML only content", result)
        self.assertNotIn("<p>", result)

    def test_plain_preferred_over_html(self):
        raw = _make_multipart_email(plain="Plain wins", html="<p>HTML loses</p>")
        msg = email.message_from_bytes(raw)
        result = _extract_body(msg)
        self.assertIn("Plain wins", result)

    def test_truncation(self):
        long_body = "A" * 2000
        raw = _make_simple_email(body=long_body)
        msg = email.message_from_bytes(raw)
        result = _extract_body(msg, max_chars=500)
        self.assertLessEqual(len(result), 600)   # truncated + suffix
        self.assertIn("truncated", result)

    def test_empty_email_returns_fallback(self):
        msg = email.message_from_bytes(b"Subject: Empty\r\n\r\n")
        result = _extract_body(msg)
        self.assertIn("no readable body", result)


class TestGetCredentials(unittest.TestCase):
    """_get_credentials reads env vars correctly."""

    def test_both_present(self):
        with patch.dict(os.environ, {"GMAIL_USER": "u@g.com", "GMAIL_APP_PASSWORD": "pass"}):
            u, p = _get_credentials()
            self.assertEqual(u, "u@g.com")
            self.assertEqual(p, "pass")

    def test_both_missing(self):
        env = {k: "" for k in ("GMAIL_USER", "GMAIL_APP_PASSWORD")}
        with patch.dict(os.environ, env, clear=False):
            # Temporarily remove them
            os.environ.pop("GMAIL_USER", None)
            os.environ.pop("GMAIL_APP_PASSWORD", None)
            u, p = _get_credentials()
            self.assertIsNone(u)
            self.assertIsNone(p)

    def test_one_missing(self):
        with patch.dict(os.environ, {"GMAIL_USER": "u@g.com"}, clear=False):
            os.environ.pop("GMAIL_APP_PASSWORD", None)
            u, p = _get_credentials()
            self.assertIsNone(u)


class TestCheckGmailDispatch(unittest.TestCase):
    """check_gmail action dispatch, validation, and error paths."""

    # ── missing credentials ────────────────────────────────────────────────

    def test_missing_credentials_returns_error_string(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GMAIL_USER", None)
            os.environ.pop("GMAIL_APP_PASSWORD", None)
            result = run(check_gmail(action="list_unread"))
            self.assertIn("Error", result)
            self.assertNotRaises_no_exception_was_raised()

    def assertNotRaises_no_exception_was_raised(self):
        pass   # just naming the intent — if we got here, no exception

    # ── unknown action ─────────────────────────────────────────────────────

    def test_unknown_action(self):
        result = run(check_gmail(action="fly_to_moon"))
        self.assertIn("unknown action", result.lower())
        self.assertIn("fly_to_moon", result)
        self.assertIn("list_unread", result)   # valid actions listed

    # ── required arg validation fires before network ───────────────────────

    def test_search_without_query(self):
        result = run(check_gmail(action="search", query=""))
        self.assertIn("Error", result)
        self.assertIn("query", result)

    def test_get_body_without_uid(self):
        result = run(check_gmail(action="get_body", uid=""))
        self.assertIn("Error", result)
        self.assertIn("uid", result)

    def test_mark_read_without_uid(self):
        result = run(check_gmail(action="mark_read", uid=""))
        self.assertIn("Error", result)
        self.assertIn("uid", result)

    # ── max_results clamping ───────────────────────────────────────────────

    @patch("tools.gmail_check._connect")
    def test_max_results_clamped_above_20(self, mock_connect):
        """Values above 20 are clamped to 20 — no IndexError or API abuse."""
        uid_list   = [str(i).encode() for i in range(1, 25)]   # 24 UIDs
        email_raw  = _make_simple_email()
        fetch_data = {uid: email_raw for uid in uid_list}
        mock_connect.return_value = _mock_imap(uid_list, fetch_data)
        with patch.dict(os.environ, {"GMAIL_USER": "u@g.com", "GMAIL_APP_PASSWORD": "p"}):
            result = run(check_gmail(action="list_unread", max_results=99))
        # Must not crash and must return at most 20 entries
        lines = [l for l in result.split("\n") if l.startswith("UID:") or "UID:" in l]
        self.assertLessEqual(len(lines), 20)

    @patch("tools.gmail_check._connect")
    def test_max_results_clamped_below_1(self, mock_connect):
        uid_list   = [b"101"]
        email_raw  = _make_simple_email()
        mock_connect.return_value = _mock_imap(uid_list, {b"101": email_raw})
        with patch.dict(os.environ, {"GMAIL_USER": "u@g.com", "GMAIL_APP_PASSWORD": "p"}):
            result = run(check_gmail(action="list_unread", max_results=0))
        self.assertNotIn("Error", result)   # 0 → clamped to 1, should still work

    # ── list_unread with mocked IMAP ──────────────────────────────────────

    @patch("tools.gmail_check._connect")
    def test_list_unread_returns_headers(self, mock_connect):
        email_raw = _make_simple_email(
            subject="Test email", sender="alice@example.com"
        )
        mock_connect.return_value = _mock_imap([b"101"], {b"101": email_raw})
        with patch.dict(os.environ, {"GMAIL_USER": "u@g.com", "GMAIL_APP_PASSWORD": "p"}):
            result = run(check_gmail(action="list_unread", max_results=5))
        self.assertIn("101", result)             # UID in output
        self.assertIn("alice@example.com", result)
        self.assertIn("Test email", result)

    @patch("tools.gmail_check._connect")
    def test_list_unread_empty_inbox(self, mock_connect):
        mock_connect.return_value = _mock_imap([], {})
        with patch.dict(os.environ, {"GMAIL_USER": "u@g.com", "GMAIL_APP_PASSWORD": "p"}):
            result = run(check_gmail(action="list_unread"))
        self.assertIn("No unread", result)

    @patch("tools.gmail_check._connect")
    def test_list_unread_most_recent_first(self, mock_connect):
        """UIDs should appear in reverse order (newest first)."""
        uid_list = [b"100", b"101", b"102"]
        fetch_data = {uid: _make_simple_email(subject=f"Email {uid.decode()}") for uid in uid_list}
        mock_connect.return_value = _mock_imap(uid_list, fetch_data)
        with patch.dict(os.environ, {"GMAIL_USER": "u@g.com", "GMAIL_APP_PASSWORD": "p"}):
            result = run(check_gmail(action="list_unread", max_results=3))
        idx_102 = result.find("102")
        idx_100 = result.find("100")
        self.assertGreater(idx_100, idx_102)   # 102 appears before 100

    # ── list_recent ───────────────────────────────────────────────────────

    @patch("tools.gmail_check._connect")
    def test_list_recent_uses_all_criteria(self, mock_connect):
        """list_recent should use ALL not UNSEEN as search criteria."""
        email_raw = _make_simple_email()
        imap      = _mock_imap([b"55"], {b"55": email_raw})
        mock_connect.return_value = imap
        with patch.dict(os.environ, {"GMAIL_USER": "u@g.com", "GMAIL_APP_PASSWORD": "p"}):
            run(check_gmail(action="list_recent", max_results=2))
        # Verify the search was called with ALL (not UNSEEN)
        calls = [str(c) for c in imap.uid.call_args_list]
        self.assertTrue(any("ALL" in c for c in calls))

    # ── count_unread ──────────────────────────────────────────────────────

    @patch("tools.gmail_check._connect")
    def test_count_unread_correct_number(self, mock_connect):
        uid_list = [b"10", b"11", b"12"]
        mock_connect.return_value = _mock_imap(uid_list, {})
        with patch.dict(os.environ, {"GMAIL_USER": "u@g.com", "GMAIL_APP_PASSWORD": "p"}):
            result = run(check_gmail(action="count_unread"))
        self.assertIn("3", result)

    @patch("tools.gmail_check._connect")
    def test_count_unread_zero(self, mock_connect):
        mock_connect.return_value = _mock_imap([], {})
        with patch.dict(os.environ, {"GMAIL_USER": "u@g.com", "GMAIL_APP_PASSWORD": "p"}):
            result = run(check_gmail(action="count_unread"))
        self.assertIn("0", result)

    # ── search ────────────────────────────────────────────────────────────

    @patch("tools.gmail_check._connect")
    def test_search_returns_matches(self, mock_connect):
        email_raw = _make_simple_email(subject="GitHub deployment failed")
        mock_connect.return_value = _mock_imap([b"200"], {b"200": email_raw})
        with patch.dict(os.environ, {"GMAIL_USER": "u@g.com", "GMAIL_APP_PASSWORD": "p"}):
            result = run(check_gmail(action="search", query="GitHub", max_results=3))
        self.assertIn("200", result)
        self.assertIn("GitHub", result)

    @patch("tools.gmail_check._connect")
    def test_search_no_matches(self, mock_connect):
        mock_connect.return_value = _mock_imap([], {})
        with patch.dict(os.environ, {"GMAIL_USER": "u@g.com", "GMAIL_APP_PASSWORD": "p"}):
            result = run(check_gmail(action="search", query="unicorn"))
        self.assertIn("No emails matched", result)

    @patch("tools.gmail_check._connect")
    def test_search_strips_quotes_from_query(self, mock_connect):
        """Quotes in query must be stripped to prevent IMAP injection."""
        email_raw = _make_simple_email()
        imap = _mock_imap([b"1"], {b"1": email_raw})
        mock_connect.return_value = imap
        with patch.dict(os.environ, {"GMAIL_USER": "u@g.com", "GMAIL_APP_PASSWORD": "p"}):
            run(check_gmail(action="search", query='say "hello"'))
        # The uid search call args should not contain unescaped double-quotes
        calls = [str(c) for c in imap.uid.call_args_list]
        search_calls = [c for c in calls if "search" in c.lower()]
        for c in search_calls:
            # The query string inside TEXT "..." must not contain raw "
            # (The quotes wrapping the TEXT value are OK — check the content)
            self.assertNotIn('say "hello"', c)

    # ── get_body ──────────────────────────────────────────────────────────

    @patch("tools.gmail_check._connect")
    def test_get_body_returns_full_content(self, mock_connect):
        raw = _make_multipart_email(plain="This is the full body content.")
        mock_connect.return_value = _mock_imap([b"300"], {b"300": raw})
        with patch.dict(os.environ, {"GMAIL_USER": "u@g.com", "GMAIL_APP_PASSWORD": "p"}):
            result = run(check_gmail(action="get_body", uid="300"))
        self.assertIn("BODY:", result)
        self.assertIn("full body content", result)

    @patch("tools.gmail_check._connect")
    def test_get_body_missing_uid_returns_error(self, mock_connect):
        mock_connect.return_value = _mock_imap([], {})
        with patch.dict(os.environ, {"GMAIL_USER": "u@g.com", "GMAIL_APP_PASSWORD": "p"}):
            result = run(check_gmail(action="get_body", uid="99999"))
        self.assertIn("Could not fetch", result)

    # ── mark_read ─────────────────────────────────────────────────────────

    @patch("tools.gmail_check._connect")
    def test_mark_read_calls_store(self, mock_connect):
        imap = _mock_imap([b"400"], {b"400": _make_simple_email()})
        mock_connect.return_value = imap
        with patch.dict(os.environ, {"GMAIL_USER": "u@g.com", "GMAIL_APP_PASSWORD": "p"}):
            result = run(check_gmail(action="mark_read", uid="400"))
        self.assertIn("400", result)
        self.assertIn("read", result.lower())
        # Verify store was called with \Seen
        calls = [str(c) for c in imap.uid.call_args_list]
        self.assertTrue(any("store" in c.lower() for c in calls))
        self.assertTrue(any("Seen" in c for c in calls))

    # ── IMAP connection failure ────────────────────────────────────────────

    @patch("tools.gmail_check._connect")
    def test_imap_login_failure_returns_error_string(self, mock_connect):
        mock_connect.return_value = None   # _connect returns None on failure
        with patch.dict(os.environ, {"GMAIL_USER": "u@g.com", "GMAIL_APP_PASSWORD": "wrongpass"}):
            result = run(check_gmail(action="list_unread"))
        self.assertIn("Error", result)
        # Must not raise
        self.assertIsInstance(result, str)

    @patch("tools.gmail_check._connect")
    def test_mid_operation_imap_error_returns_error_string(self, mock_connect):
        """An IMAP error mid-operation must return a string, never raise."""
        imap = MagicMock(spec=imaplib.IMAP4_SSL)
        imap.select.return_value = ("OK", [b"3"])
        imap.uid.side_effect = imaplib.IMAP4.error("connection reset")
        imap.logout.return_value = ("BYE", [b""])
        mock_connect.return_value = imap
        with patch.dict(os.environ, {"GMAIL_USER": "u@g.com", "GMAIL_APP_PASSWORD": "p"}):
            result = run(check_gmail(action="list_unread"))
        self.assertIsInstance(result, str)
        self.assertIn("Failed", result)

    # ── non-ASCII headers ─────────────────────────────────────────────────

    @patch("tools.gmail_check._connect")
    def test_non_ascii_subject_decoded_correctly(self, mock_connect):
        """RFC 2047 encoded subjects (common with Indian senders) must not crash."""
        import base64
        encoded = base64.b64encode("नमस्ते Kairos".encode("utf-8")).decode()
        subject_header = f"=?utf-8?b?{encoded}?="

        raw = _make_simple_email(subject=subject_header)
        mock_connect.return_value = _mock_imap([b"500"], {b"500": raw})
        with patch.dict(os.environ, {"GMAIL_USER": "u@g.com", "GMAIL_APP_PASSWORD": "p"}):
            result = run(check_gmail(action="list_unread"))
        self.assertIsInstance(result, str)
        self.assertIn("Kairos", result)   # at minimum the ASCII part appears


# ── integration tests (skipped without real credentials) ─────────────────────

HAVE_CREDS = bool(
    os.getenv("GMAIL_USER") and os.getenv("GMAIL_APP_PASSWORD")
)

@unittest.skipUnless(HAVE_CREDS, "GMAIL_USER and GMAIL_APP_PASSWORD not set — skipping integration tests")
class TestCheckGmailIntegration(unittest.TestCase):
    """
    Real IMAP calls — skipped unless env vars are present.

    Run manually:
        GMAIL_USER=you@gmail.com GMAIL_APP_PASSWORD=xxxx pytest test/test_gmail_check.py -v -k integration
    """

    def test_count_unread_is_integer_string(self):
        result = run(check_gmail(action="count_unread"))
        self.assertIsInstance(result, str)
        self.assertNotIn("Error", result)
        # Result must contain a number
        import re
        self.assertTrue(re.search(r"\d+", result), f"No number in: {result}")

    def test_list_unread_returns_string(self):
        result = run(check_gmail(action="list_unread", max_results=3))
        self.assertIsInstance(result, str)
        self.assertNotIn("Error", result)

    def test_list_recent_returns_string(self):
        result = run(check_gmail(action="list_recent", max_results=3))
        self.assertIsInstance(result, str)
        self.assertNotIn("Error", result)

    def test_search_returns_string(self):
        # Search for something that almost certainly exists
        result = run(check_gmail(action="search", query="Google", max_results=3))
        self.assertIsInstance(result, str)
        # Either found results or "No emails matched" — both are valid
        self.assertTrue(
            "No emails matched" in result or "Search results" in result,
            f"Unexpected output: {result[:200]}"
        )

    def test_get_body_with_real_uid(self):
        """
        Get a real UID from list_unread, then fetch its body.
        Skipped gracefully if inbox is empty.
        """
        list_result = run(check_gmail(action="list_unread", max_results=1))
        if "No unread" in list_result:
            self.skipTest("Inbox is empty — cannot test get_body")

        import re
        uids = re.findall(r"UID:\s*(\d+)", list_result)
        self.assertTrue(uids, f"No UID found in list output: {list_result}")

        body_result = run(check_gmail(action="get_body", uid=uids[0]))
        self.assertIsInstance(body_result, str)
        self.assertIn("FROM:", body_result)
        self.assertIn("BODY:", body_result)
        self.assertNotIn("Error", body_result)


if __name__ == "__main__":
    unittest.main(verbosity=2)