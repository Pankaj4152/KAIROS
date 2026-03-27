"""
Telegram channel adapter for Kairos.

Responsibilities:
  - Receive messages from Telegram via long-polling
  - Authenticate — reject any user who isn't you (checked by numeric user ID)
  - Normalize into KairosEvent and pass to the orchestrator
  - Collect the full streamed response and send it back
  - Split long responses at natural boundaries (paragraph → sentence → hard cut)

Why Telegram collects the full response before sending:
  Telegram's sendMessage is not a streaming API. Editing a message repeatedly
  to simulate streaming is flickery and rate-limited. Better UX: show typing
  indicator, then send the complete response as one (or a few) messages.
  Voice is where we stream token-by-token — different channel, different output.

Security model:
  The bot is publicly discoverable on Telegram. TELEGRAM_USER_ID is the only
  thing standing between your tasks/memory/API credits and anyone who finds it.
  Numeric ID is used (not username) because usernames can change.
"""

import asyncio
import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters
from config.settings import TELEGRAM_MAX_LENGTH, TYPING_REFRESH_SECS, MIN_MESSAGE_INTERVAL
load_dotenv()

from gateway.normalizer import normalize_telegram
from orchestrator.orchestrator import orchestrator

logger = logging.getLogger(__name__)



class TelegramChannel:

    def __init__(self):
        self.token           = os.getenv("TELEGRAM_BOT_TOKEN")
        self.allowed_user_id = int(os.getenv("TELEGRAM_USER_ID", "0"))

        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set in .env")
        if not self.allowed_user_id:
            raise ValueError("TELEGRAM_USER_ID not set in .env")

        self._last_message_time: float = 0.0   # simple rate limiting

        self.app = Application.builder().token(self.token).build()
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

    # ─── auth ─────────────────────────────────────────────────────────────────

    def _is_authorized(self, update: Update) -> bool:
        """Check numeric user ID — not username, which can change."""
        return (
            update.message is not None
            and update.message.from_user.id == self.allowed_user_id
        )

    # ─── typing indicator ─────────────────────────────────────────────────────

    async def _keep_typing(self, update: Update, stop_event: asyncio.Event) -> None:
        """
        Send typing action every TYPING_REFRESH_SECS until stop_event is set.
        Runs as a background task alongside orchestrator.process().
        Telegram typing indicators expire after ~5s — we refresh before that.
        """
        while not stop_event.is_set():
            try:
                await update.message.chat.send_action(ChatAction.TYPING)
            except Exception:
                pass   # don't let a failed typing action crash the handler
            await asyncio.sleep(TYPING_REFRESH_SECS)

    # ─── response splitting ───────────────────────────────────────────────────

    def _split_response(self, text: str) -> list[str]:
        """
        Split a response into Telegram-sized chunks at natural boundaries.

        Order of preference:
            1. Paragraph breaks (\n\n) — most natural
            2. Hard cut at TELEGRAM_MAX_LENGTH — last resort for huge paragraphs

        Why not truncate: the user asked a question, they deserve the full answer.
        """
        if len(text) <= TELEGRAM_MAX_LENGTH:
            return [text]

        chunks: list[str] = []
        current = ""

        for para in text.split("\n\n"):
            candidate = f"{current}\n\n{para}".strip() if current else para

            if len(candidate) <= TELEGRAM_MAX_LENGTH:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Single paragraph larger than limit — hard split
                if len(para) > TELEGRAM_MAX_LENGTH:
                    for i in range(0, len(para), TELEGRAM_MAX_LENGTH):
                        chunks.append(para[i : i + TELEGRAM_MAX_LENGTH])
                    current = ""
                else:
                    current = para

        if current:
            chunks.append(current)

        return chunks

    # ─── main handler ─────────────────────────────────────────────────────────

    async def _handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Entry point for every incoming Telegram message.

        Flow:
            1. Guard — validate update has a message
            2. Auth  — reject silently if not the allowed user
            3. Rate  — ignore if too soon after last message
            4. Type  — start persistent typing indicator
            5. Run   — collect full response from orchestrator
            6. Send  — split and reply
        """
        # Guard — edited messages, channel posts, etc. may have no .message
        if not update.message or not update.message.text:
            return

        # Auth — silent reject, no feedback to the unauthorized user
        if not self._is_authorized(update):
            logger.warning("Rejected unauthorized user: %s", update.message.from_user.id)
            return

        # Rate limit — ignore rapid duplicate sends
        import time
        now = time.time()
        if now - self._last_message_time < MIN_MESSAGE_INTERVAL:
            logger.debug("Rate limited message from %s", update.message.from_user.id)
            return
        self._last_message_time = now

        # Typing indicator — runs in background while we generate
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(self._keep_typing(update, stop_typing))

        try:
            event = normalize_telegram(update)

            # Collect full response — Telegram can't receive a streaming push
            tokens: list[str] = []
            async for token in orchestrator.process(event):
                tokens.append(token)

            response_text = "".join(tokens)

        except Exception as e:
            logger.exception("Orchestrator failed for Telegram message: %s", e)
            response_text = "Something went wrong. Try again in a moment."

        finally:
            # Always stop the typing indicator, even if orchestrator raised
            stop_typing.set()
            typing_task.cancel()

        # Send — one message per chunk
        chunks = self._split_response(response_text)
        for chunk in chunks:
            try:
                await update.message.reply_text(chunk)
            except Exception as e:
                logger.error("Failed to send Telegram message: %s", e)
                break   # if one chunk fails, don't try to send the rest

    # ─── lifecycle ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Start the Telegram bot. Runs until cancelled (e.g. Ctrl+C or SIGTERM).
        Initialises the orchestrator before polling starts so the first message
        never takes the profile-load hit.
        """
        logger.info("Kairos Telegram channel starting...")

        # Load orchestrator profile before first message arrives
        await orchestrator.startup()

        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram: polling started")

        try:
            await asyncio.Event().wait()
        finally:
            logger.info("Telegram: shutting down...")
            stop_typing_tasks = [
                t for t in asyncio.all_tasks()
                if t.get_name().startswith("Task-")   # best-effort cleanup
            ]
            for t in stop_typing_tasks:
                t.cancel()
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()