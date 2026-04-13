"""
Enhanced debugging utilities for tracing LLM calls and request flow.
Set LOG_LEVEL=TRACE in .env to enable verbose logging.
"""

import logging
import json
from typing import Any

logger = logging.getLogger(__name__)

# Map log levels
TRACE = 5
logging.addLevelName(TRACE, "TRACE")

def trace(msg: str, *args, **kwargs):
    """Ultra-verbose tracing for deep debugging."""
    if logger.isEnabledFor(TRACE):
        logger.log(TRACE, msg, *args, **kwargs)

def debug_payload(name: str, payload: dict, max_chars: int = 500):
    """Log request/response payloads nicely."""
    if logger.isEnabledFor(logging.DEBUG):
        json_str = json.dumps(payload, indent=2)[:max_chars]
        logger.debug(f"[{name}] {json_str}...")

def debug_messages(stage: str, messages: list[dict]):
    """Log message list for debugging."""
    if logger.isEnabledFor(logging.DEBUG):
        msg_summary = [
            f"{m['role']}: {str(m['content'])[:80]}"
            for m in messages
        ]
        logger.debug(f"[{stage}] Messages: {', '.join(msg_summary)}")