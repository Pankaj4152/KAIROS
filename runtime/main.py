"""
Kairos entry point.

Startup sequence:
    1. init_db()             — SQLite tables (tasks, events, habits, spending)
    2. init_vector_store()   — sqlite-vec virtual table + memory_meta
    3. orchestrator.startup() — load profile.md into memory
    4. Run channels          — telegram + webui via asyncio.gather()

Adding a new channel:
    1. Build it in channels/
    2. Import and instantiate it here
    3. Add channel.run() to the gather() call
    Nothing else changes.
"""

import asyncio
import logging
import os
import sys


from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger


sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # Add project root for config module

from dotenv import load_dotenv
load_dotenv()

from config.logging_config import setup_logging
setup_logging()
logger = logging.getLogger("kairos")

async def morning_briefing():
    """
    Daily briefing pushed to Telegram at configured time.
    Fetches tasks + events, composes a summary, sends it proactively.

    Why Telegram and not WebUI:
        WebUI needs the browser open. Telegram delivers to your phone
        even when you're asleep — that's the point of a proactive agent.
    """
    import os
    from memory.sqlite_store import fetch_open_tasks, fetch_upcoming_events
    from llm.client import LLMClient
    from telegram import Bot

    logger.info("Running morning briefing...")

    try:
        tasks  = await fetch_open_tasks()
        events = await fetch_upcoming_events(limit=5)

        task_lines = "\n".join(
            f"- [{t['priority']}] {t['title']}"
            f"{' (due ' + t['due_date'] + ')' if t['due_date'] else ''}"
            for t in tasks[:5]
        ) or "No open tasks."

        event_lines = "\n".join(
            f"- {e['title']} at {e['start_time']}"
            for e in events
        ) or "Nothing scheduled."

        prompt = f"""
            You are Kairos, a personal AI assistant giving a morning briefing.
            Be concise and energetic. No markdown. Plain text only. 3-4 sentences max.

            Open tasks:
            {task_lines}

            Upcoming events:
            {event_lines}

            Give a brief morning summary — what to focus on today and any events coming up.
            """

        llm     = LLMClient()
        message = await llm.complete(
            [{"role": "user", "content": prompt}],
            tier=2,
            timeout=30.0,
        )

        bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
        await bot.send_message(
            chat_id=os.getenv("TELEGRAM_USER_ID"),
            text=f"Good morning!\n\n{message}",
        )
        logger.info("Morning briefing sent.")

    except Exception as e:
        logger.error("Morning briefing failed: %s", e)


async def main() -> None:
    # Import here — after sys.path and load_dotenv are set up
    from memory.sqlite_store import init_db
    from memory.vector_store import init_vector_store
    from orchestrator.orchestrator import orchestrator
    from channels.telegram import TelegramChannel
    from channels.webui import WebUIChannel

    logger.info("Kairos starting...")

    # 1. Structured store — tasks, events, habits, spending, schema_version
    init_db()
    logger.info("Database ready")

    # 2. Vector store — must come after init_db() because both use kairos.db
    #    init_vector_store() creates memory_vec and memory_meta if missing
    init_vector_store()
    logger.info("Vector store ready")

    # 3. Data directories — sessions dir must exist before first write
    data_dir     = os.getenv("DATA_DIR", "./data")
    sessions_dir = os.path.join(data_dir, "sessions")
    os.makedirs(sessions_dir, exist_ok=True)
    logger.info("Data directories ready: %s", data_dir)

    # 4. Orchestrator profile — load once, shared by all channels
    await orchestrator.startup()
    logger.info("Orchestrator ready")

    # Scheduler — morning briefing
    BRIEFING_HOUR   = int(os.getenv("BRIEFING_HOUR", "7"))
    BRIEFING_MINUTE = int(os.getenv("BRIEFING_MINUTE", "30"))

    scheduler = AsyncIOScheduler(timezone=os.getenv("TIMEZONE", "Asia/Kolkata"))
    scheduler.add_job(
        morning_briefing,
        CronTrigger(hour=BRIEFING_HOUR, minute=BRIEFING_MINUTE),
        id="morning_briefing",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "Briefing scheduled at %02d:%02d %s",
        BRIEFING_HOUR, BRIEFING_MINUTE,
        os.getenv("TIMEZONE", "Asia/Kolkata")
    )

    # 5. Channels
    telegram = TelegramChannel()
    webui    = WebUIChannel()

    logger.info("Kairos is live")
    logger.info("  WebUI    → http://localhost:8000")
    logger.info("  Telegram → message your bot")
    logger.info("Press Ctrl+C to stop")

    # Both channels run forever. If one crashes the exception propagates
    # to asyncio.run() and you see the full traceback — good for dev.
    await asyncio.gather(
        telegram.run(),
        webui.run(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Kairos stopped")