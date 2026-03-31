"""
Async write-back — runs after every response, never blocks the user.

Three jobs run in parallel after each completed exchange:
    1. embed_turn   — embed the conversation turn into sqlite-vec
    2. append_session — write both turns to the session JSON file
    3. extract_facts  — use tier-1 LLM to find structured facts → SQLite

Called via asyncio.create_task() from orchestrator.py:
    asyncio.create_task(run_writeback(session_id, user_text, response_text, ...))

The user never waits for any of this. If writeback fails, the response
was already delivered — log the error and move on.

Fact extractor:
    Modular — any LLM client can be injected.
    Currently uses tier-1 (phi-3-mini) — fast, free, good enough for
    extracting simple structured facts from conversation turns.
    Swap by passing a different llm_client to run_writeback().
"""

import asyncio
import json
import logging

from llm.client import LLMClient
from memory.session_store import append_turn, needs_compaction, compact, get_history
from memory.vector_store import embed_and_store

logger = logging.getLogger(__name__)


# ─── fact extractor prompt ────────────────────────────────────────────────────

# Runs on tier-1 (local phi-3-mini) — must return JSON only.
# Use str.replace for injection — not .format() — user text may contain { }.
_FACT_PROMPT = """\
Extract structured facts from this conversation turn.
Return ONLY a JSON array. No explanation, no markdown, nothing else.
If no facts found, return exactly: []

Format: [{"type": "preference|fact|goal", "key": "...", "value": "..."}]

Only extract facts EXPLICITLY stated by the user. Do not infer.

USER: REPLACE_USER
ASSISTANT: REPLACE_ASSISTANT
"""


# ─── individual jobs ─────────────────────────────────────────────────────────

async def _embed_turn(
    session_id: str,
    user_text: str,
    response_text: str,
) -> None:
    """
    Embed the full conversation turn (user + assistant) into sqlite-vec.

    We embed both sides together as one unit — this captures the context
    of the exchange, not just the question or just the answer.
    """
    content = f"USER: {user_text}\nASSISTANT: {response_text}"
    try:
        row_id = await embed_and_store(
            content=content,
            source="conversation",
            session_id=session_id,
        )
        logger.debug("Embedded turn → memory_vec rowid=%d", row_id)
    except Exception as e:
        # Embedding failure is non-fatal — session history still works
        logger.warning("embed_turn failed for session %s: %s", session_id[:8], e)


async def _append_session(
    session_id: str,
    user_text: str,
    response_text: str,
    channel: str,
    tier: int,
    llm: LLMClient,
) -> None:
    try:
        await append_turn(session_id, "user", user_text,
                    meta={"channel": channel})
        await append_turn(session_id, "assistant", response_text,
                    meta={"tier_used": tier})

        if await needs_compaction(session_id):
            logger.info("Session %s needs compaction", session_id[:8])
            await _compact_session(session_id, llm)

    except Exception as e:
        logger.warning("append_session failed for %s: %s", session_id[:8], e)


async def _compact_session(session_id: str, llm: LLMClient) -> None:
    try:
        history = await get_history(session_id, last_n=20)
        if not history:
            return
        history_text = "\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in history
        )
        prompt = (
            "Summarise this conversation in 2-3 sentences. "
            "Focus on facts, decisions, and context useful for continuing later.\n\n"
            f"{history_text}"
        )
        summary = await llm.complete(
            [{"role": "user", "content": prompt}],
            tier=1,
            timeout=15.0,
        )
        await compact(session_id, summary)
        logger.info("Session %s compacted", session_id[:8])

    except Exception as e:
        logger.warning("compact_session failed for %s: %s", session_id[:8], e)

async def _extract_facts(
    user_text: str,
    response_text: str,
    llm: LLMClient,
) -> list[dict]:
    """
    Use tier-1 LLM to extract structured facts from the conversation turn.
    Returns a list of fact dicts. Returns [] on any failure — never raises.

    Extracted facts are written to preferences.json.
    Future: route specific fact types to SQLite (e.g. spending facts → spending table).
    """
    prompt = (
        _FACT_PROMPT
        .replace("REPLACE_USER", user_text)
        .replace("REPLACE_ASSISTANT", response_text)
    )

    try:
        raw = await llm.complete(
            [{"role": "user", "content": prompt}],
            tier=1,
            timeout=10.0,
        )

        # Parse defensively — tier-1 models are inconsistent
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]).strip()

        start = text.find("[")
        end   = text.rfind("]") + 1
        if start == -1 or end == 0:
            return []

        facts = json.loads(text[start:end])
        if not isinstance(facts, list):
            return []

        # Validate each fact has required keys and string values
        valid = []
        for f in facts:
            if (
                isinstance(f, dict)
                and all(k in f for k in ("type", "key", "value"))
                and all(isinstance(f[k], str) for k in ("type", "key", "value"))
                and f["type"] in ("preference", "fact", "goal")
            ):
                valid.append(f)

        return valid

    except Exception as e:
        logger.warning("extract_facts failed: %s", e)
        return []


async def _save_facts(facts: list[dict], data_dir: str) -> None:
    """
    Persist extracted facts to preferences.json.

    Merges into existing prefs — new facts overwrite existing keys
    of the same type. This means Kairos learns and updates over time:
    if you said you prefer light coffee last month and now say dark,
    the dark preference wins.
    """
    import os
    import asyncio

    if not facts:
        return

    prefs_path = os.path.join(data_dir, "preferences.json")

    def _run():
        # Load existing prefs
        try:
            with open(prefs_path, "r", encoding="utf-8") as f:
                prefs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            prefs = {"preferences": {}, "facts": {}, "goals": {}}

        # Merge — keyed by (type, key) so updates overwrite old values
        for fact in facts:
            section = fact["type"] + "s"   # "preference" → "preferences"
            if section not in prefs:
                prefs[section] = {}
            prefs[section][fact["key"]] = fact["value"]

        with open(prefs_path, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=2, ensure_ascii=False)

        logger.debug("Saved %d facts to preferences.json", len(facts))

    await asyncio.to_thread(_run)

# Intents that carry no useful information for memory.
# Session history is still preserved (conversation flow matters),
# but embedding + fact extraction are skipped to avoid noise.
SKIP_MEMORY_INTENTS = frozenset({"chitchat"})


# ─── main entry point ─────────────────────────────────────────────────────────

async def run_writeback(
    session_id: str,
    user_text: str,
    response_text: str,
    channel: str,
    tier: int,
    intent: str = "question",
    llm: LLMClient | None = None,
    data_dir: str | None = None,
) -> None:
    """
    Run write-back jobs after a completed response.

    Call this via asyncio.create_task() — never await it directly.
    The user already has their response. This runs in the background.

    For low-information intents (chitchat), only session append runs.
    Embedding and fact extraction are skipped to avoid memory pollution.

    Args:
        session_id:    which session to write to.
        user_text:     the user's message.
        response_text: Kairos's full response.
        channel:       "voice" | "telegram" | "webui" | "cron".
        tier:          which model tier handled this response (1/2/3).
        intent:        classified intent — used to skip memory for chitchat.
        llm:           LLM client for compaction + fact extraction.
                       Defaults to a new LLMClient() if not provided.
        data_dir:      where preferences.json lives. Defaults to DATA_DIR env var.
    """
    import os
    _llm      = llm or LLMClient()
    _data_dir = data_dir or os.getenv("DATA_DIR", "./data")

    skip_memory = intent in SKIP_MEMORY_INTENTS

    logger.debug(
        "Writeback starting for session %s (intent=%s, skip_memory=%s)",
        session_id[:8], intent, skip_memory,
    )

    # Session append always runs — preserves conversation flow
    session_task = _append_session(session_id, user_text, response_text, channel, tier, _llm)

    if skip_memory:
        # Chitchat — only save session history, skip embedding + facts
        await session_task
        logger.debug(
            "Writeback complete for session %s (skipped embed+facts for %s)",
            session_id[:8], intent,
        )
        return

    # Full writeback — embed + session + facts in parallel
    embed_task = _embed_turn(session_id, user_text, response_text)
    facts_task = _extract_facts(user_text, response_text, _llm)

    embed_result, _, facts = await asyncio.gather(
        embed_task,
        session_task,
        facts_task,
        return_exceptions=True,
    )

    # Log any job-level exceptions — individual jobs also catch internally
    if isinstance(embed_result, Exception):
        logger.warning("embed job raised: %s", embed_result)
    if isinstance(facts, Exception):
        logger.warning("facts job raised: %s", facts)
        facts = []

    # Save facts if any were extracted
    if facts:
        logger.info(
            "Extracted %d fact(s) from session %s: %s",
            len(facts), session_id[:8], facts,
        )
        await _save_facts(facts, _data_dir)

    logger.debug("Writeback complete for session %s", session_id[:8])