# Mark 2 Release Notes

Date: 2026-04-16

## Summary

Mark 2 focuses on reliability and routing correctness.
The release adds stronger fallback behavior in the orchestrator, improves classifier behavior and latency paths, updates model routing, and refreshes documentation to match runtime behavior.

## Highlights

### 1) Resilience Improvements

- Stream fallback across tiers (3 -> 2 -> 1) when generation fails.
- Tool-loop fallback behavior per round to avoid hard request failure.
- Safe session-history wrapper so history read failures do not block responses.
- Max tool-round degradation path with safe user-facing fallback message.

Primary code areas:
- runtime/orchestrator/orchestrator.py
- runtime/orchestrator/classifier.py
- runtime/llm/client.py

### 2) Routing and Latency Updates

- Classifier prompt/rules tuned for better routing decisions.
- Tier mapping/config updated in LiteLLM config.
- Warmup defaults updated for local tiers to reduce first-request cold-start penalties.
- Added direct latency probe utility for per-tier benchmarking.

Primary code areas:
- runtime/orchestrator/classifier.py
- litellm/config.yaml
- runtime/llm/client.py
- runtime/latency_probe.py

### 3) Messaging + UX

- Added messaging tool integration and registry updates.
- Web UI received light-theme updates.

Primary code areas:
- runtime/tools/messaging.py
- runtime/tools/registry.py
- runtime/static/index.html

### 4) Documentation Sync

- README, Setup, Guide, Architecture markdown, and architecture HTML were aligned to current model/config behavior.
- Environment reference updated for current keys and search backends.

Primary docs:
- README.md
- docs/SETUP.md
- docs/GUIDE.md
- docs/ARCHITECTURE.md
- docs/RESILIENCE.md
- docs/architecture.html
- RELEASES.md
- .env.example

## Operational Notes

- Ensure LiteLLM routes match tier aliases tier1, tier2, tier3 in litellm/config.yaml.
- Ensure .env includes GEMINI_API_KEY (tier 3), plus optional backend keys such as SERPER_API_KEY when using Serper.
- Run local model warmup on startup to reduce cold starts.

## Diagnostics

Latency probe example:

```bash
python runtime/latency_probe.py --models tier1 tier2 tier3
```

## Backward Compatibility

- No database migration required in this release.
- Existing sessions/data remain compatible.

## Known Gaps

- Tool-level circuit-breaker and per-tool retry knobs are now configuration-ready but not fully wired into executor behavior yet.

## Recommended Post-Release Verification

```bash
pytest -q
```

If failures occur, check:
- LiteLLM proxy availability and tier routing
- Ollama availability for local tiers
- Environment variable completeness in .env
