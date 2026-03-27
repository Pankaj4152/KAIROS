"""
Tests for runtime/gateway/normalizer.py

Covers:
  - normalize_voice: channel, modality, text stripping
  - normalize_webui: channel, modality, raw source
  - normalize_cron: fresh UUID session, job_id in raw
  - normalize_telegram: extracts user_id, chat_id from mock Update
  - Empty text raises ValueError for all normalizers
"""

import logging
import time
from unittest.mock import MagicMock

import pytest

logger = logging.getLogger(__name__)


# ── voice ─────────────────────────────────────────────────────────────────────

class TestNormalizeVoice:
    def test_basic_fields(self):
        from gateway.normalizer import normalize_voice
        event = normalize_voice("Hello Kairos", session_id="voice-001")
        logger.info("Voice event: channel=%s modality=%s text=%r", event.channel, event.modality, event.text)
        assert event.channel == "voice"
        assert event.modality == "voice"
        assert event.text == "Hello Kairos"
        assert event.session_id == "voice-001"

    def test_strips_whitespace(self):
        from gateway.normalizer import normalize_voice
        event = normalize_voice("  Hello  ", session_id="v-002")
        logger.info("Stripped text: %r", event.text)
        assert event.text == "Hello"

    def test_empty_text_raises(self):
        from gateway.normalizer import normalize_voice
        with pytest.raises(ValueError, match="Empty text"):
            normalize_voice("", session_id="v-003")
        logger.info("Empty voice text correctly rejected")

    def test_whitespace_only_raises(self):
        from gateway.normalizer import normalize_voice
        with pytest.raises(ValueError, match="Empty text"):
            normalize_voice("   ", session_id="v-004")
        logger.info("Whitespace-only voice text correctly rejected")

    def test_has_timestamp(self):
        from gateway.normalizer import normalize_voice
        before = time.time()
        event = normalize_voice("Test", session_id="v-005")
        after = time.time()
        logger.info("Timestamp: %.2f (between %.2f and %.2f)", event.timestamp, before, after)
        assert before <= event.timestamp <= after


# ── webui ─────────────────────────────────────────────────────────────────────

class TestNormalizeWebui:
    def test_basic_fields(self):
        from gateway.normalizer import normalize_webui
        event = normalize_webui("Show my tasks", session_id="webui-001")
        logger.info("WebUI event: channel=%s modality=%s", event.channel, event.modality)
        assert event.channel == "webui"
        assert event.modality == "text"
        assert event.text == "Show my tasks"
        assert event.raw == {"source": "websocket"}

    def test_empty_text_raises(self):
        from gateway.normalizer import normalize_webui
        with pytest.raises(ValueError, match="Empty text"):
            normalize_webui("", session_id="w-002")
        logger.info("Empty webui text correctly rejected")


# ── cron ──────────────────────────────────────────────────────────────────────

class TestNormalizeCron:
    def test_session_id_starts_with_cron(self):
        from gateway.normalizer import normalize_cron
        event = normalize_cron("Run morning briefing")
        logger.info("Cron session_id: %s", event.session_id)
        assert event.session_id.startswith("cron-")
        assert event.channel == "cron"

    def test_each_call_gets_unique_session(self):
        from gateway.normalizer import normalize_cron
        e1 = normalize_cron("Briefing 1")
        e2 = normalize_cron("Briefing 2")
        logger.info("Session IDs: %s vs %s", e1.session_id, e2.session_id)
        assert e1.session_id != e2.session_id

    def test_job_id_in_raw(self):
        from gateway.normalizer import normalize_cron
        event = normalize_cron("Briefing", job_id="morning_brief")
        logger.info("Raw: %s", event.raw)
        assert event.raw["job_id"] == "morning_brief"

    def test_empty_text_raises(self):
        from gateway.normalizer import normalize_cron
        with pytest.raises(ValueError, match="Empty text"):
            normalize_cron("")
        logger.info("Empty cron text correctly rejected")


# ── telegram ──────────────────────────────────────────────────────────────────

class TestNormalizeTelegram:
    def _mock_update(self, text="Hello", chat_id=12345, user_id=67890, update_id=1):
        """Build a mock python-telegram-bot Update object."""
        update = MagicMock()
        update.update_id = update_id
        update.message.text = text
        update.message.caption = None
        update.message.chat_id = chat_id
        update.message.from_user.id = user_id
        update.message.date.timestamp.return_value = 1700000000.0
        return update

    def test_extracts_fields(self):
        from gateway.normalizer import normalize_telegram
        update = self._mock_update(text="What's the weather?", chat_id=111, user_id=222)
        event = normalize_telegram(update)
        logger.info(
            "Telegram event: text=%r user_id=%s session_id=%s channel=%s",
            event.text, event.user_id, event.session_id, event.channel,
        )
        assert event.text == "What's the weather?"
        assert event.user_id == "222"
        assert event.session_id == "111"
        assert event.channel == "telegram"
        assert event.modality == "text"

    def test_uses_caption_when_no_text(self):
        from gateway.normalizer import normalize_telegram
        update = self._mock_update()
        update.message.text = None
        update.message.caption = "Photo caption"
        event = normalize_telegram(update)
        logger.info("Caption fallback: %r", event.text)
        assert event.text == "Photo caption"

    def test_empty_text_raises(self):
        from gateway.normalizer import normalize_telegram
        update = self._mock_update(text="")
        update.message.caption = None
        with pytest.raises(ValueError, match="Empty text"):
            normalize_telegram(update)
        logger.info("Empty telegram text correctly rejected")
