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

import json
import logging
import re
import time

from llm.client import LLMClient

logger = logging.getLogger(__name__)


# ─── fast-path pattern ────────────────────────────────────────────────────────

# Matches greetings and trivial casual phrases. Checked BEFORE any LLM call.
# Intentionally conservative — only catches clear greetings.
# Everything else still goes through the LLM classifier.
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

_CHITCHAT_RESULT: dict = {
    "intent":       "chitchat",
    "complexity":   1,
    "domains":      [],
    "tools_needed": [],
    "tier":         1,
}


# ─── valid values ─────────────────────────────────────────────────────────────

# Clamp classifier output to these sets — never trust raw model output blindly.
# Unknown values are silently dropped, not passed downstream.
VALID_DOMAINS = frozenset({"tasks", "events", "habits", "spending", "memory"})
VALID_TOOLS   = frozenset({"web_search", "send_message"})
VALID_INTENTS = frozenset({"question", "task", "reminder", "memory",
                           "chitchat", "search", "code"})


# ─── defaults ─────────────────────────────────────────────────────────────────

# Returned on any classifier failure. tier=1 is the safe default:
# if the classifier can't even parse the input, it's likely simple enough
# for the local model. Truly complex queries produce parseable JSON.
DEFAULT_RESULT: dict = {
    "intent":       "question",
    "complexity":   1,
    "domains":      [],
    "tools_needed": [],
    "tier":         1,
}


# ─── prompt ───────────────────────────────────────────────────────────────────

# {message} is replaced with str.replace() — not .format() — so curly braces
# in user messages (code, JSON, etc.) don't corrupt the prompt.
CLASSIFIER_PROMPT = """\
You are a request classifier. Output ONLY raw JSON. No text. No markdown.

JSON shape:
{"intent":"<intent>","complexity":<1-3>,"domains":[],"tools_needed":[],"tier":<1-3>}

CRITICAL RULES (follow strictly):
1. Greetings and casual messages are ALWAYS intent=chitchat, complexity=1, tier=1
2. Simple factual questions are complexity=1, tier=1
3. Web search, task management are complexity=2, tier=2
4. Code writing, reasoning, research, multi-step planning are complexity=3, tier=3

Valid intents: question, task, reminder, memory, chitchat, search, code

Valid domains (include ONLY if relevant, otherwise empty):
- "tasks" — todos, work items
- "events" — meetings, schedule
- "habits" — streaks, routines
- "spending" — money, expenses
- "memory" — recall past conversations

Valid tools_needed (otherwise empty):
- "web_search" — current info, news, prices, weather
- "send_message" — send a message

Examples:
"hello" -> {"intent":"chitchat","complexity":1,"domains":[],"tools_needed":[],"tier":1}
"hey" -> {"intent":"chitchat","complexity":1,"domains":[],"tools_needed":[],"tier":1}
"good morning" -> {"intent":"chitchat","complexity":1,"domains":[],"tools_needed":[],"tier":1}
"how are you" -> {"intent":"chitchat","complexity":1,"domains":[],"tools_needed":[],"tier":1}
"thanks" -> {"intent":"chitchat","complexity":1,"domains":[],"tools_needed":[],"tier":1}
"what tasks do i have" -> {"intent":"question","complexity":1,"domains":["tasks"],"tools_needed":[],"tier":1}
"what's the weather" -> {"intent":"search","complexity":2,"domains":[],"tools_needed":["web_search"],"tier":2}
"search for latest AI news" -> {"intent":"search","complexity":2,"domains":[],"tools_needed":["web_search"],"tier":2}
"write a python script to scrape a website" -> {"intent":"code","complexity":3,"domains":[],"tools_needed":[],"tier":3}

Classify this: {message}
"""


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
            return DEFAULT_RESULT.copy()

        try:
            result = json.loads(text[start:end])
        except json.JSONDecodeError as e:
            logger.warning("Classifier: JSON parse failed: %s | raw: %r", e, raw[:100])
            return DEFAULT_RESULT.copy()

        # Build output — start from defaults, overwrite with valid parsed values
        final = DEFAULT_RESULT.copy()

        # Scalar fields — coerce to correct types
        if "intent" in result and result["intent"] in VALID_INTENTS:
            final["intent"] = result["intent"]

        if "complexity" in result:
            final["complexity"] = max(1, min(3, int(result["complexity"])))

        if "tier" in result:
            final["tier"] = max(1, min(3, int(result["tier"])))

        # List fields — filter to known valid values only
        # Unknown strings from phi3 are silently dropped, never passed downstream
        if "domains" in result and isinstance(result["domains"], list):
            final["domains"] = [
                d for d in result["domains"] if d in VALID_DOMAINS
            ]

        if "tools_needed" in result and isinstance(result["tools_needed"], list):
            final["tools_needed"] = [
                t for t in result["tools_needed"] if t in VALID_TOOLS
            ]

        return final

    async def classify(self, text: str) -> dict:
        """
        Classify a user message. Always returns a valid routing dict.

        Fast path:
            Regex catches obvious greetings/casual phrases → instant return.
            No LLM call, no latency, no cost.

        Slow path:
            Everything else → tier-1 LLM (phi) classification.

        Returns:
            intent       — what the user wants
            complexity   — 1/2/3 (depth of response needed)
            domains      — which SQLite stores to query
            tools_needed — which tools to invoke before the LLM call
            tier         — which model to use (1=local, 2=haiku, 3=sonnet)
        """
        # Fast path — regex catches trivial greetings before any LLM call
        if _CHITCHAT_RE.match(text):
            logger.info(
                "Classified %r → chitchat (fast path, 0.00s)", text[:40],
            )
            return _CHITCHAT_RESULT.copy()

        # Slow path — LLM-based classification
        # Use str.replace, not .format() — user text may contain { } characters
        prompt = CLASSIFIER_PROMPT.replace("{message}", text)
        t0 = time.perf_counter()

        try:
            raw = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=1,
                timeout=10.0,  # classifier must be fast — hard limit
            )
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
            return DEFAULT_RESULT.copy()


# ─── singleton ────────────────────────────────────────────────────────────────

classifier = Classifier()