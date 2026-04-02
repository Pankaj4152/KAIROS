"""
Messaging tool handler.
Allows the LLM to proactively push messages to the user via Telegram.
"""

import logging
import os
from telegram import Bot

logger = logging.getLogger(__name__)

async def send_message(message: str) -> str:
    """
    Sends a message to the user via the Telegram Bot API.
    Enforces a strict 4096 character limit so the LLM learns to chunk.
    Returns success or detailed error string to the LLM.
    """
    if len(message) > 4096:
        return (
            "Error: Message too long. Telegram max is 4096 characters. "
            f"Your message was {len(message)} characters. Please shorten it "
            "or intelligently chunk it into multiple separate tool calls (e.g. 'Part 1 of 2')."
        )

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_USER_ID")
    
    if not token or not chat_id:
        return "Error: TELEGRAM_BOT_TOKEN or TELEGRAM_USER_ID is missing from environment variables."

    try:
        # Spin up a localized Bot instance to fire and forget to Telegram
        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=message)
        logger.info("send_message tool successfully dispatched %d characters", len(message))
        return "SUCCESS: Message sent to Telegram. You must now reply to the user in text confirming it was sent, and DO NOT call this tool again."
    except Exception as e:
        logger.warning("send_message tool failed: %s", e)
        return f"Error sending message to Telegram API: {str(e)}"
