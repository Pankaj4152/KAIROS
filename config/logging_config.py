"""
Centralized logging configuration for Kairos.

Call setup_logging() once at startup (main.py) to configure all loggers.
Individual modules keep using `logger = logging.getLogger(__name__)` as before.

Env vars:
    LOG_LEVEL  — root log level (default: INFO)
    LOG_FILE   — optional path for file logging (e.g. ./data/kairos.log)
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from config.settings import LOG_FORMAT, LOG_DATE_FORMAT, LOG_MAX_BYTES, LOG_BACKUP_COUNT


# Third-party loggers that are too noisy at DEBUG/INFO
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "apscheduler",
    "urllib3",
    "asyncio",
    "websockets",
)


def setup_logging(level_override: str | None = None) -> None:
    """
    Configure the root logger with console + optional file handlers.

    Args:
        level_override: Force a specific level (used by tests to force DEBUG).
                        If None, reads LOG_LEVEL from env (default: INFO).
    """
    level_name = level_override or os.getenv("LOG_LEVEL", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)

    root = logging.getLogger()

    # Avoid adding duplicate handlers if called more than once
    if root.handlers:
        root.handlers.clear()

    root.setLevel(level)

    # ── Console handler ───────────────────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    root.addHandler(console)

    # ── File handler (optional) ───────────────────────────────────────────
    log_file = os.getenv("LOG_FILE", "").strip()
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        root.addHandler(file_handler)

    # ── Quiet noisy third-party loggers ───────────────────────────────────
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger("kairos").debug(
        "Logging configured: level=%s, file=%s", level_name.upper(), log_file or "(console only)"
    )
