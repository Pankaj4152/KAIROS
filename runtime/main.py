"""
Kairos entry point.

Startup sequence:
    1. init_db()           — create SQLite tables if they don't exist
    2. orchestrator.startup() — load profile.md into memory
    3. Run all channels    — telegram + webui run forever via asyncio.gather()

Adding a new channel:
    1. Build it in channels/
    2. Import and instantiate it here
    3. Add channel.run() to the gather() call
    Nothing else needs to change.
"""

import asyncio
import logging
import os
import sys

# Allow imports from the runtime/ directory without installing as a package
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from memory.sqlite_store import init_db
from orchestrator.orchestrator import orchestrator
from channels.telegram import TelegramChannel
from channels.webui import WebUIChannel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("kairos")


async def main() -> None:
    """
    Wire everything together and run all channels concurrently.

    All channels are async and spend most of their time waiting for
    input — asyncio.gather() runs them in one event loop with no
    threads, no locks, no shared-state issues.
    """
    logger.info("Kairos starting...")

    # 1. Database — idempotent, safe every restart
    init_db()
    logger.info("Database ready")

    # 2. Ensure data directories exist before any channel needs them
    data_dir     = os.getenv("DATA_DIR", "./data")
    sessions_dir = os.path.join(data_dir, "sessions")
    os.makedirs(sessions_dir, exist_ok=True)
    logger.info("Data directories ready: %s", data_dir)

    # 3. Load profile once here — both channels share the same orchestrator
    #    singleton, so this covers all channels with one file read.
    await orchestrator.startup()
    logger.info("Orchestrator ready")

    # 4. Build channels
    telegram = TelegramChannel()
    webui    = WebUIChannel()

    logger.info("Kairos is live")
    logger.info("  WebUI    → http://localhost:8000")
    logger.info("  Telegram → message your bot")
    logger.info("Press Ctrl+C to stop")

    # Run all channels forever.
    # If one crashes, gather() will cancel the other and the exception
    # propagates to asyncio.run() — you'll see the full traceback.
    await asyncio.gather(
        telegram.run(),
        webui.run(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # asyncio.run() handles loop cleanup — we just suppress the traceback
        logger.info("Kairos stopped")