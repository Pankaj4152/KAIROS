# Resilience Architecture

> How KAIROS keeps responding when components fail.

**<- [Back to README](../README.md)** · [Architecture](ARCHITECTURE.md) · [Setup](SETUP.md)

---

## Overview

KAIROS includes multi-layer resilience so request handling degrades gracefully under failures.
The goal is continuity and safe user output, not hard failure.

---

## Tier Fallback

- Streaming path: Tier 3 -> Tier 2 -> Tier 1
- Tool-calling path: per-round tier fallback
- Context/history fetch: graceful fallback to empty context/history when read fails
- Max tool rounds: degrade to plain generation when tool loop limit is reached

---

## Error Handling

- Transient LLM/API errors: retry with exponential backoff
- Non-retryable/request-shape errors: fail fast for that attempt
- Tool input errors: schema reject and continue safely
- Terminal failure path: return user-safe fallback message

### Current vs Configured-Next Knobs

The following are available as resilience configuration knobs:

- `TOOL_CALL_MAX_RETRIES=2`
- `TOOL_CALL_RETRY_BASE_DELAY=0.5`
- `TOOL_CIRCUIT_BREAKER_FAILURE_THRESHOLD=3`
- `TOOL_CIRCUIT_BREAKER_COOLDOWN_SECONDS=60`
- `MAX_TOOL_ROUNDS=5`

Note:
- `MAX_TOOL_ROUNDS` is actively used by the orchestrator tool loop.
- Circuit-breaker and tool-call retry knobs are now configuration-ready and can be wired into tool-executor logic in a follow-up patch.

---

## Recovery Flow (ASCII)

```
request -> classify -> preferred tier
                    |
                    v
                 try tier
                    |
          +---------+---------+
          |                   |
       success             failure
          |                   |
       stream            try lower tier
          |                   |
          +----------- until exhausted -----------+
                                                   |
                                                   v
                                     return safe fallback message
```

---

## Where It Lives In Code

- `runtime/orchestrator/orchestrator.py` (tier fallback, max-round degradation, safe fallback message)
- `runtime/orchestrator/classifier.py` (safe default classification fallback)
- `runtime/llm/client.py` (retry/backoff for transient failures)
- `runtime/tools/executor.py` (schema validation, timeout-guarded execution)
- `config/settings.py` (resilience config constants)
