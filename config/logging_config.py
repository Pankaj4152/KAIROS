"""
Centralized logging configuration for Kairos.

Call setup_logging() once at startup (main.py) to configure all loggers.
Individual modules keep using `logger = logging.getLogger(__name__)` as before.

Env vars:
    LOG_LEVEL          — root log level (default: INFO)
    LOG_CONSOLE_LEVEL  — console-only override (default: same as LOG_LEVEL)
    LOG_FILE           — optional path for file logging (e.g. ./data/kairos.log)
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from config.settings import LOG_FORMAT, LOG_DATE_FORMAT, LOG_MAX_BYTES, LOG_BACKUP_COUNT


# Third-party loggers that are too noisy at DEBUG/INFO.
# These are set to WARNING so they don't flood the log with internal details.
_NOISY_LOGGERS = (
    # HTTP clients
    "httpx",
    "httpcore",
    "urllib3",
    # Async / scheduling
    "apscheduler",
    "asyncio",
    "websockets",
    # Telegram — polling generates 3 DEBUG lines every 10s
    "telegram.ext.ExtBot",
    "telegram.ext.Updater",
    "telegram.ext",
    "telegram.ext.Application",
    # Web search HTTP internals (TLS handshakes, HTTP/2 frames, cookies)
    "primp",
    "primp.connect",
    "rustls",
    "rustls.client.hs",
    "rustls.client.tls13",
    "h2",
    "h2.client",
    "h2.codec.framed_write",
    "h2.codec.framed_read",
    "h2.frame.settings",
    "h2.proto.connection",
    "h2.proto.settings",
    "h2.hpack.decoder",
    "hyper_util",
    "hyper_util.client.legacy.connect.http",
    "hyper_util.client.legacy.pool",
    "cookie_store",
    "cookie_store.cookie_store",
    # Misc
    "tzlocal",
)


_TRACE_LEVEL = 5


def _resolve_level(level_name: str, default_level: int) -> int:
    """Resolve log level names, including custom TRACE."""
    upper = level_name.upper()
    if upper == "TRACE":
        return _TRACE_LEVEL
    return getattr(logging, upper, default_level)


def setup_logging(level_override: str | None = None) -> None:
    """
    Configure the root logger with console + optional file handlers.

    Args:
        level_override: Force a specific level (used by tests to force DEBUG).
                        If None, reads LOG_LEVEL from env (default: INFO).
    """
    level_name = level_override or os.getenv("LOG_LEVEL", "INFO")
    level = _resolve_level(level_name, logging.INFO)

    # Console can run at a higher level than the file to keep terminal clean
    console_level_name = os.getenv("LOG_CONSOLE_LEVEL", level_name).upper()
    console_level = _resolve_level(console_level_name, level)

    root = logging.getLogger()

    # Avoid adding duplicate handlers if called more than once
    if root.handlers:
        root.handlers.clear()

    # Root must be set to the lowest of all handler levels
    root.setLevel(min(level, console_level))

    # ── Console handler ───────────────────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(console_level)
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
        "Logging configured: level=%s, console=%s, file=%s",
        level_name.upper(), console_level_name, log_file or "(console only)",
    )
