"""
Classifier — routes every request to the right tier, domains, and tools.

Runs on tier-1 (phi3 local, ~100ms, free) before any cloud LLM call.
Returns a routing dict the orchestrator uses to assemble context and
choose the right model.

Why local phi3 and not Haiku:
  This runs on every single request. Using a cloud model here means
  paying for two cloud calls per message. phi3 is free and fast enough
  for intent classification even if it's weak at reasoning.

Failure contract:
  classify() never raises. On any failure — model timeout, bad JSON,
  missing keys — it returns DEFAULT_RESULT (tier=2, no domains, no tools).
  The user gets a slightly more expensive response. Never a crash.
"""

import asyncio
import json
import logging
import re
import time

from config.settings import CLASSIFIER_TIMEOUT
from llm.client import LLMClient
from llm.debug import trace

from tools.registry import get_eligibility, REGISTRY


logger = logging.getLogger(__name__)


# ─── fast-path pattern ────────────────────────────────────────────────────────
_CHITCHAT_RE = re.compile(
    r"^\s*"
    r"(h(i|ey|ello|owdy|ola)|yo|sup|hey+|what'?s\s*up"
    r"|good\s*(morning|afternoon|evening|night)"
    r"|how\s*(are|r)\s*(you|u|ya)"
    r"|thanks?|thank\s*you|thx|ty"
    r"|bye|goodbye|see\s*ya|later|gn|good\s*night"
    r"|ok(ay)?|cool|nice|great|awesome|lol|haha|hmm+"
    r"|yes|no|yep|nope|yeah|nah"
    r")\s*[!?.]*\s*$",
    re.IGNORECASE,
)

def _chitchat_result() -> dict:
    return {
        "intent":               "chitchat",
        "complexity":           1,
        "domains":              [],
        "needs_external_tools": False, # Isko Boolean kar diya prompt ke mutabik
        "tools_needed":         [],
        "tier":                 1,
    }



# ─── Dynamic validation values ───────────────────────────────────────────────

VALID_INTENTS = frozenset({"question", "task", "reminder", "memory", "chitchat", "search", "code"})

def get_valid_tools() -> set[str]:
    """Registry se dynamically check karta hai kaunse tools eligible aur enabled hain."""
    return set(name for name, eligible in get_eligibility().items() if eligible)

def get_valid_domains() -> set[str]:
    """Registry se dynamically check karta hai kaunse domains valid hain SQLite memory + tools ke liye."""
    db_domains = {"tasks", "events", "habits", "spending", "memory"}
    tool_domains = {tool["domain"] for tool in REGISTRY.values() if tool.get("domain")}
    return db_domains | tool_domains




# ─── Dynamic Defaults ─────────────────────────────────────────────────────────

def _default_result() -> dict:
    return {
        "intent":               "question",
        "complexity":           1,
        "domains":              [],
        "needs_external_tools": False,
        "tools_needed":         [],
        "tier":                 2,
    }


DEFAULT_RESULT = _default_result()


# ─── Dynamic Prompt Generator ────────────────────────────────────────────────

def get_classifier_prompt() -> str:
    """Registry ke valid domains ko dynamically prompt me inject karega."""
    domains_str = ", ".join(get_valid_domains())
    return f"""\
You are a request classifier. Output ONLY raw JSON. No text. No markdown.

JSON shape:
{{"intent":"<intent>","complexity":<1-3>,"domains":["<generic_domains>"],"needs_external_tools":true/false,"tier":<1-3>}}

CRITICAL RULES:
1. If the user asks for real-time info, news, weather, calendar updates, or any action outside your core logic -> needs_external_tools=true, tier=2
2. Code writing, deep research, and multi-step planning are complexity=3, tier=3

Valid intents: question, task, reminder, memory, chitchat, search, code
Valid domains: {domains_str}

Classify this: {{message}}"""









# ─── classifier ───────────────────────────────────────────────────────────────

class Classifier:
    """
    Runs on tier-1 (local phi3) before any cloud call.
    Single responsibility: text → routing dict.
    Injecting llm_client makes it trivially testable.
    """

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    def _parse(self, raw: str) -> dict:
        """
        Parse classifier JSON output defensively.

        phi3 sometimes:
          - Wraps output in ```json fences despite being told not to
          - Adds a sentence before the JSON object
          - Returns string values where ints are expected ("tier": "2")
          - Hallucinates unknown domain or tool names

        All of these are handled here. If parsing fails entirely,
        DEFAULT_RESULT is returned.
        """
        trace("Classifier.parse start raw_len=%d raw_preview=%r", len(raw), raw[:200])
        text = raw.strip()

        # Strip markdown fences if phi3 ignored the prompt instruction
        if text.startswith("```"):
            lines = text.split("\n")
            text  = "\n".join(lines[1:-1]).strip()

        # Find the JSON object — handles preamble sentences
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning("Classifier: no JSON found in: %r", raw[:100])
            return _default_result()


        # try to convert the JSON substring into a dict. If this fails, return defaults.
        try:
            result = json.loads(text[start:end])
        except json.JSONDecodeError as e:
            logger.warning("Classifier: JSON parse failed: %s | raw: %r", e, raw[:100])
            trace("Classifier.parse fallback=default reason=json_decode_error error=%s", e)
            return _default_result()



        # Build output — start from defaults, overwrite with valid parsed values
        final = _default_result()

        # Scalar fields — coerce to correct types
        if "intent" in result and result["intent"] in VALID_INTENTS:
            final["intent"] = result["intent"]

        if "complexity" in result:
            final["complexity"] = max(1, min(3, int(result["complexity"])))

        if "tier" in result:
            final["tier"] = max(1, min(3, int(result["tier"])))

        if "needs_external_tools" in result:
            final["needs_external_tools"] = bool(result["needs_external_tools"])

        if "domains" in result and isinstance(result["domains"], list):
            valid_domains = get_valid_domains()
            final["domains"] = [d for d in result["domains"] if d in valid_domains]

        # List fields — filter to known valid values only, or dynamically construct
        if "tools_needed" in result and isinstance(result["tools_needed"], list):
            valid_tools = get_valid_tools()
            final["tools_needed"] = [
                t for t in result["tools_needed"] if t in valid_tools
            ]
            if final["tools_needed"]:
                final["needs_external_tools"] = True
        elif final.get("needs_external_tools"):
            tools = []
            valid_tools = get_valid_tools()
            if final.get("intent") == "search" and "web_search" in valid_tools:
                tools.append("web_search")
            for t in valid_tools:
                if t != "web_search":
                    tools.append(t)
            final["tools_needed"] = tools
        else:
            final["tools_needed"] = []

        return final

    async def classify(self, text: str, trace_id: str | None = None, metadata: dict | None = None) -> dict:
        """
        Classify a user message. Always returns a valid routing dict.

        Fast path:
            Regex catches obvious greetings/casual phrases → instant return.
            No LLM call, no latency, no cost.

        Slow path:
            Everything else → tier-2 LLM classification.

        Returns:
            intent       — what the user wants
            complexity   — 1/2/3 (depth of response needed)
            domains      — which SQLite stores to query
            tools_needed — which tools to invoke before the LLM call
            tier         — which model to use
        """
        # Fast path — regex catches trivial greetings before any LLM call
        if _CHITCHAT_RE.match(text):
            return _chitchat_result()

        # Slow path — LLM-based classification
        # Use str.replace, not .format() — user text may contain { } characters
        prompt = get_classifier_prompt().replace("{message}", text)
        t0 = time.perf_counter()
        

        try:
            # Set a timeout for the classifier call from centralized settings.
            complete_kwargs = {
                "messages": [{"role": "user", "content": prompt}],
                "tier": 1,
                "timeout": CLASSIFIER_TIMEOUT,
                "trace_id": trace_id,
            }
            if metadata is not None:
                complete_kwargs["metadata"] = metadata

            raw = await self.llm.complete(**complete_kwargs)
            result = self._parse(raw)
            logger.info(
                "Classified %r → %s (%.2fs)",
                text[:40], result, time.perf_counter() - t0,
            )
            return result

        except Exception as e:
            logger.warning(
                "Classifier failed, using default (%.2fs): %s",
                time.perf_counter() - t0, e,
            )
            return _default_result()



classifier = Classifier()