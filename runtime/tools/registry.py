"""
Tool registry — single source of truth for every tool Kairos can call.

Each entry declares:
    description     what the tool does (shown to LLM)
    schema          JSON schema for input validation (enforced before dispatch)
    handler         async callable — must accept **kwargs matching schema props
    enabled         False = never dispatched, even if LLM requests it
    requires_env    list of env vars that must be set for this tool to be eligible
                    checked at startup via check_eligibility(), not at call time

Adding a new tool:
    1. Write the handler in tools/<name>.py
    2. Add an entry here
    3. That's it — executor picks it up automatically
"""

import logging
import os

logger = logging.getLogger(__name__)


# ── lazy imports ───────────────────────────────────────────────────────────────
# Import inside functions so a broken tool file doesn't kill the whole registry

def _load_web_search():
    from tools.web_search import web_search
    return web_search



# ── registry ───────────────────────────────────────────────────────────────────

REGISTRY: dict[str, dict] = {

    "web_search": {
        "description": (
            "Search the web for current information. Use when the user asks about "
            "recent events, facts you may not know, or anything requiring up-to-date data."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                    "description": "The search query string",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "handler": _load_web_search,   # callable that returns the actual handler
        "enabled": True,
        "requires_env": [],            # duckduckgo default needs no key
    },

}


# ── eligibility check ──────────────────────────────────────────────────────────

def check_eligibility() -> dict[str, bool]:
    """
    Called once at startup. Logs a warning for every tool that is registered
    but cannot run due to missing env vars. Returns eligibility map.

    Executor uses this map — ineligible tools are rejected before handler load.
    """
    eligibility: dict[str, bool] = {}

    for name, tool in REGISTRY.items():
        if not tool["enabled"]:
            eligibility[name] = False
            logger.debug("Tool '%s' is disabled", name)
            continue

        missing = [var for var in tool["requires_env"] if not os.getenv(var)]
        if missing:
            eligibility[name] = False
            logger.warning(
                "Tool '%s' ineligible — missing env vars: %s", name, missing
            )
        else:
            eligibility[name] = True
            logger.info("Tool '%s' ready", name)

    return eligibility


# Module-level eligibility map — populated on first import
_eligibility: dict[str, bool] | None = None

def get_eligibility() -> dict[str, bool]:
    global _eligibility
    if _eligibility is None:
        _eligibility = check_eligibility()
    return _eligibility


def get_tool_schemas() -> list[dict]:
    """
    Returns tool descriptions in the shape LiteLLM / Anthropic expects.
    Only includes enabled + eligible tools.
    Used by context assembler to tell the LLM what tools exist.
    """
    eligibility = get_eligibility()
    schemas = []
    for name, tool in REGISTRY.items():
        if not eligibility.get(name):
            continue
        schemas.append({
            "name": name,
            "description": tool["description"],
            "input_schema": tool["schema"],
        })
    return schemas