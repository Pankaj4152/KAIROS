import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from memory.sqlite_store import init_db
from channels.telegram import TelegramChannel
from channels.webui import WebUIChannel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("kairos")


async def main():
    """
    Entry point. Starts all channels simultaneously.

    Why asyncio.gather():
        All three channels are async — they sit idle waiting for
        input most of the time. gather() runs them concurrently
        in one event loop. One process, one Python interpreter,
        all channels live at the same time.

        This is different from threading — no shared state issues,
        no locks needed. asyncio channels cooperate by yielding
        control when waiting for I/O.

    Startup order:
        1. init_db() — create tables if they don't exist (safe, idempotent)
        2. Start all channels — they each run forever until Ctrl+C
    """
    logger.info("Kairos starting...")

    # Init database — safe to call every startup
    init_db()
    logger.info("Database ready.")

    # Build channels
    telegram = TelegramChannel()
    webui = WebUIChannel()

    logger.info("All channels initialised. Kairos is live.")
    logger.info("  WebUI:    http://localhost:8000")
    logger.info("  Telegram: message your bot")
    logger.info("Press Ctrl+C to stop.")

    # Run all channels forever, concurrently
    try:
        await asyncio.gather(
            telegram.run(),
            webui.run(),
        )
    except KeyboardInterrupt:
        logger.info("Kairos shutting down.")


if __name__ == "__main__":
    asyncio.run(main())