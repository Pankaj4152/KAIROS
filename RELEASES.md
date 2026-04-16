# Version History

## v2.0.0 - Resilient Intelligence (April 2026)

- Multi-tier automatic fallback (tier 3 -> tier 2 -> tier 1)
- Tool-loop degradation when max rounds are exhausted
- Safe session-history wrapper for corrupted/unavailable history
- Classifier fallback to safe default route on parse/timeout failure
- LLM retry/backoff for transient failures
- Updated docs for architecture, setup, and operational diagnostics

Notes:
- This release uses current tier aliases (`tier1`, `tier2`, `tier3`) from `litellm/config.yaml`.
- Circuit-breaker and tool-call retry knobs are configuration-ready and can be fully wired in subsequent iterations.

## v1.0.0 - Functional Baseline (March 2026)

- Core request pipeline (gateway -> classifier -> context -> stream -> writeback)
- Telegram and WebUI channels
- Task/event/habit/spending memory domains
- Web search integration with pluggable backends
- Voice pipeline marked as in progress
