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

from llm.client import LLMClient

logger = logging.getLogger(__name__)


# ─── valid values ─────────────────────────────────────────────────────────────

# Clamp classifier output to these sets — never trust raw model output blindly.
# Unknown values are silently dropped, not passed downstream.
VALID_DOMAINS = frozenset({"tasks", "events", "habits", "spending", "memory"})
VALID_TOOLS   = frozenset({"web_search", "calendar_write", "send_message"})
VALID_INTENTS = frozenset({"question", "task", "reminder", "memory",
                           "chitchat", "search", "code"})


# ─── defaults ─────────────────────────────────────────────────────────────────

# Returned on any classifier failure. tier=2 is the safe middle ground:
# fast enough for voice, capable enough for most requests.
DEFAULT_RESULT: dict = {
    "intent":       "question",
    "complexity":   2,
    "domains":      [],
    "tools_needed": [],
    "tier":         2,
}


# ─── prompt ───────────────────────────────────────────────────────────────────

# {message} is replaced with str.replace() — not .format() — so curly braces
# in user messages (code, JSON, etc.) don't corrupt the prompt.
CLASSIFIER_PROMPT = """\
You are a request classifier for a personal AI assistant.
Respond ONLY with valid JSON. No explanation. No markdown. No code blocks. Just the raw JSON object.

Classify this request: {message}

Return exactly this JSON shape:
{
  "intent": "<question|task|reminder|memory|chitchat|search|code>",
  "complexity": <1|2|3>,
  "domains": [],
  "tools_needed": [],
  "tier": <1|2|3>
}

Rules for complexity and tier:
- complexity 1, tier 1: trivial (greetings, time, simple facts, list my tasks)
- complexity 2, tier 2: moderate (calendar queries, web search, task management)
- complexity 3, tier 3: complex (reasoning, code writing, research, multi-step planning)

Rules for domains (which memory stores to query — include only what is relevant):
- "tasks"    — todos, work items, what to do
- "events"   — calendar, meetings, schedule, today, tomorrow
- "habits"   — streaks, routines, daily check-ins
- "spending" — money, expenses, budget, purchases
- "memory"   — remember, recall, what did i say, earlier

Rules for tools_needed (tools to invoke before the LLM response):
- "web_search"     — current information, news, prices, weather
- "calendar_write" — creating or editing a calendar event
- "send_message"   — send a message to someone

Examples:
- "hey" -> {"intent":"chitchat","complexity":1,"domains":[],"tools_needed":[],"tier":1}
- "what tasks do i have" -> {"intent":"question","complexity":1,"domains":["tasks"],"tools_needed":[],"tier":1}
- "search for latest AI news" -> {"intent":"search","complexity":2,"domains":[],"tools_needed":["web_search"],"tier":2}
- "write a python script to scrape a website" -> {"intent":"code","complexity":3,"domains":[],"tools_needed":[],"tier":3}
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

        Returns:
            intent       — what the user wants
            complexity   — 1/2/3 (depth of response needed)
            domains      — which SQLite stores to query
            tools_needed — which tools to invoke before the LLM call
            tier         — which model to use (1=local, 2=haiku, 3=sonnet)
        """
        # Use str.replace, not .format() — user text may contain { } characters
        prompt = CLASSIFIER_PROMPT.replace("{message}", text)

        try:
            raw = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                tier=1,
                timeout=10.0,  # classifier must be fast — hard limit
            )
            result = self._parse(raw)
            logger.debug("Classified %r → %s", text[:40], result)
            return result

        except Exception as e:
            logger.warning("Classifier failed, using default: %s", e)
            return DEFAULT_RESULT.copy()


# ─── singleton ────────────────────────────────────────────────────────────────

classifier = Classifier()