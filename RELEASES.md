# Version History

## v3.0.0 - Mark 3: Integrated Automation & Proactive Briefings (June 2026)

*Detailed release notes, code changes, and test verification can be found in the [Mark 3 Release Notes](file:///f:/Projects/KAIROS/docs/RELEASE_NOTES_MARK3.md) document.*

- **Email Channel briefings**: Daily morning briefings are now sent proactively via secure HTML emails using Gmail SMTP transport (replacing Telegram alerts).
- **Gmail IMAP Tool**: Added `check_gmail` tool for extracting unread inbox mail headers asynchronously.
- **Google Calendar Tool**: Full list/search/create/update/delete support for Google Calendar.
- **Latency Tester Benchmarks**: Upgraded `latency_probe.py` with multi-scenario benchmarks (`simple`, `reasoning`, `essay`, `system_design`, `coding`) and concurrent load options.
- **Dynamic Registry & Classification**: Refined Classifier prompts and schemas to resolve valid tool and domain mappings dynamically.
- **Voice dependencies cleanup**: Commented out voice pipeline libraries (`pyaudio`, `pipecat-ai`) from primary `requirements.txt` to avoid cross-platform compilation errors during initial base setups. Voice remains ongoing/under-development.

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
