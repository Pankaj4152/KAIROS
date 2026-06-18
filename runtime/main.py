"""
Kairos entry point.

Startup sequence:
    1. init_db()              — SQLite tables (tasks, events, habits, spending)
    2. init_vector_store()    — sqlite-vec virtual table + memory_meta
    3. orchestrator.startup() — load profile.md into memory
    4. Warm up Ollama models  — prevents cold start latency
    5. Start APScheduler     — schedules morning briefings
    6. Run channels           — telegram + webui via asyncio.gather()

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
from dotenv import load_dotenv

from utils.storage import storage_manager


# Ensure local imports work cleanly
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # Add project root for config module

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
    from memory.sqlite_store import fetch_open_tasks, fetch_upcoming_events
    from llm.client import LLMClient
    # from channels.telegram import Bot

    logger.info("Running morning briefing...")

    try:
        tasks  =  fetch_open_tasks()
        events =  fetch_upcoming_events(limit=5)

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

        # bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        # user_id = os.getenv("TELEGRAM_USER_ID")

        # if not bot_token or not user_id:
        #     logger.error("Skipping briefing: TELEGRAM_BOT_TOKEN or TELEGRAM_USER_ID not configured.")
        #     return

        # # Use an async context manager if your Bot client supports it, 
        # # or invoke via the shared runner context to prevent unclosed connection leaks.
        # bot = Bot(token=bot_token)
        # await bot.send_message(
        #     chat_id=user_id,
        #     text=f"Good morning!\n\n{message}",
        # )
        # Trigger Gmail client delivery
        from channels.email import EmailChannel
        gmail_channel = EmailChannel()
        await gmail_channel.send_briefing(
            subject="⏳ Your KAIROS Morning Briefing", 
            content=message
        )
        logger.info("Morning briefing sent successfully.")

    except Exception as e:
        logger.error("Morning briefing failed: %s", e, exc_info=True)

async def run_channel_safely(name: str, channel_coro):
    """Runs a communication channel and catches fatal errors to keep others alive."""
    logger.info(f"Starting channel: {name}")
    try:
        await channel_coro
    except asyncio.CancelledError:
        logger.info(f"Channel {name} was requested to stop.")
    except Exception as e:
        logger.error(f"CRITICAL ERROR in channel [{name}]: {e}", exc_info=True)
        logger.warning(f"Channel [{name}] has gone offline, but KAIROS core remains alive.")
        
        # Optional: Keep trying to restart it every 30 seconds if it crashes
        # await asyncio.sleep(30)
        # await run_channel_safely(name, channel_coro)


async def main() -> None:
    # Explicit imports deferred until environment paths are fully prepared
    from memory.sqlite_store import init_db
    from memory.vector_store import init_vector_store
    from orchestrator.orchestrator import orchestrator
    from channels.telegram import TelegramChannel
    from channels.webui import WebUIChannel

    logger.info("Kairos starting...")

    # 1. Pull down base core files first
    critical_files = ["kairos.db", "preferences.json", "profile.md"]
    await storage_manager.sync_down(critical_files)
    # 2. Pull down past session json history files
    await storage_manager.sync_down_sessions()


    # 1. Structured store — tasks, events, habits, spending, schema_version
    init_db()
    logger.info("Database ready")

    # 2. Vector store — must come after init_db() because both share kairos.db
    init_vector_store()
    logger.info("Vector store ready")

    # 3. Data directories — sessions dir must exist before first memory write
    data_dir = os.getenv("DATA_DIR", "./data")
    sessions_dir = os.path.join(data_dir, "sessions")
    os.makedirs(sessions_dir, exist_ok=True)
    logger.info("Data directories ready: %s", data_dir)

    # 4. Orchestrator profile — load profile.md once to share across channels
    #    Diagnostics & Warmup
    from channels.email import EmailChannel
    email_checker = EmailChannel()
    
    # We await it, but we DO NOT throw an exception or exit if it returns False
    email_ready = await email_checker.verify_transport()
    
    if not email_ready:
        logger.warning("KAIROS will start up, but morning briefings via email might fail.")

    await orchestrator.startup()
    logger.info("Orchestrator ready")

   # 5. Warm up local Ollama models — keeps qwen tier 1 and tier 2 hot
    #    so your first live morning command doesn't hit a cold-start latency lag
    await orchestrator.llm.warmup()
    logger.info("LLM models warmed up")

    # 6. Background Scheduler Setup
    BRIEFING_HOUR   = int(os.getenv("BRIEFING_HOUR", "7"))
    BRIEFING_MINUTE = int(os.getenv("BRIEFING_MINUTE", "30"))
    tz_str = os.getenv("TIMEZONE", "Asia/Kolkata")

    scheduler = AsyncIOScheduler(timezone=tz_str)
    scheduler.add_job(
        morning_briefing,
        CronTrigger(hour=BRIEFING_HOUR, minute=BRIEFING_MINUTE),
        id="morning_briefing",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "Briefing scheduled at %02d:%02d %s",
        BRIEFING_HOUR, BRIEFING_MINUTE, tz_str
    )

    # 5. Channels
    telegram = TelegramChannel()
    webui    = WebUIChannel()

    logger.info("Kairos is live")
    logger.info("  WebUI    → http://localhost:8000")
    logger.info("  Telegram → message your bot")
    logger.info("Press Ctrl+C to stop")

    # Wrap both channel runners safely
    await asyncio.gather(
        run_channel_safely("Telegram", telegram.run()),
        run_channel_safely("WebUI", webui.run()),
        return_exceptions=True  # Prevents one failure from stopping the gather call
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Kairos stopped cleanly via user interrupt.")
    except Exception as fatal_err:
        logger.critical("Kairos encountered a critical crash on initialization: %s", fatal_err, exc_info=True)
        sys.exit(1)