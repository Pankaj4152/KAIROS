# Mark 3 — Integrated Automation, Cloud Sync & Observability

Date: June 19, 2026 | Stability: Production-Ready

---

## What's New

### 📬 Proactive Email Briefings
* **Gmail SMTP Channel**: Daily morning briefings are now formatted into responsive, clean HTML structures and delivered proactively via secure Gmail SMTP transport. This replaces the old Telegram alerts format for daily briefings, landing directly in your inbox.
* **Startup Transport Handshake**: Integrated automated SMTP connection validation on startup, preventing silent failures of scheduled morning runs before the scheduler kicks off.

### 📧 Gmail IMAP Integration
* **`check_gmail` Tool**: A dedicated read-only tool for extracting inbox headers, searching for messages by keyword, retrieving full message bodies via stable IMAP UIDs, and updating read/unread flags.

### 📅 Full Google Calendar Write/Update Support
* **Multi-Action Event Lifecycle**: Advanced calendar integration from read-only to full event lifecycle management. You can now prompt Kairos to:
  * Create new events (`create_event`) with titles, timings, and descriptions.
  * Reschedule or modify existing events (`update_event`).
  * Remove events by exact ID or keyword search (`delete_event`).
  * Full support for timezone preservation and Google Calendar ID tracking.

### ☁️ Cloud Backup Sync (Cloudflare R2)
* **Zero-Loss Data Sync**: Integrates a custom R2 storage manager (`R2StorageManager`) using `boto3`. At startup, it syncs down critical files (`kairos.db`, `preferences.json`, `profile.md`) and all past conversation session files from a Cloudflare R2 bucket.
* **Background Sync-Up**: During session writebacks, updates are synchronized back up to R2 in the background using thread pool task runners (`sync_up_background`).

### 📊 Langfuse Tracing & Metrics
* **Observed Execution Trails**: Fully integrated **Langfuse** into the orchestrator pipeline. It initiates execution traces at the start of a query and propagates `langfuse_trace_id` metadata throughout the streaming LLM calls, fallback loops, and tool execution rounds.
* **Telemetry Profiling**: Provides basic tracking for token consumption, model latencies, tool execution successes, and cost analysis. Further improvements (such as custom feedback tags and prompt versioning) will be rolled out next.

### ⚙️ LiteLLM Model Routing & Deployed Proxy
* **Deployed Proxy Architecture**: Transitioned LiteLLM to a fully deployed proxy environment. This allows KAIROS to take advantage of advanced proxy features such as centralized model routing, load balancing, rate limiting, and api key telemetry.
* **Tier Proxy Mapping**: Centralizes system configurations by routing model requests (`tier1`, `tier2`, `tier3`) through the LiteLLM proxy gateway, mapping them dynamically to standard cloud and local backends.

### 📈 Upgraded Latency Benchmarking
* **`latency_probe.py` Upgrades**: Extended the benchmarking runner to support multiple load profiles (`simple`, `reasoning`, `essay`, `system_design`, `coding`) and concurrent query patterns, making operational latency profiling highly predictable under load.

---

## Enhancements
* **Dynamic Tool Registry**: The orchestrator now uses a dynamic registry schema in [registry.py](/runtime/tools/registry.py) to resolve valid tools, validating inputs against schemas before execution.
* **Classifier Prompt Refinement**: Rewritten classifier logic with structured guidelines to prevent local routing errors and hallucination of tool inputs.
* **Clean Base Setup**: Commented out compilation-heavy audio libraries (`pyaudio`, `pipecat-ai`) from the base `requirements.txt` to guarantee cross-platform environment setup in under 2 minutes.

---

## Codebase Changes

### Core Orchestration
* [runtime/main.py](/runtime/main.py): Scheduled APScheduler morning briefings to route through the secure Gmail SMTP transport, adding error boundaries to ensure communication channel failures never crash the runtime.
* [runtime/tools/registry.py](runtime/tools/registry.py): Integrated new tool schemas for `check_gmail` and `gmail_actions`.
* [runtime/orchestrator/orchestrator.py](/runtime/orchestrator/orchestrator.py): Integrated Langfuse observation tracking and trace propagation attributes.
* [runtime/utils/storage.py](/runtime/utils/storage.py) `[NEW]`: Implements the `R2StorageManager` interface to manage remote R2 backups.

### Tools & API Wrappers
* [runtime/tools/gmail_actions.py](/runtime/tools/gmail_actions.py) `[NEW]`: Handles send, reply, forward, delete, archive, move, mark_unread, list_folders, and create_draft using secure IMAP/SMTP connections.
* [runtime/tools/gmail_check.py](/runtime/tools/gmail_check.py) `[NEW]`: Handles read-only check tasks including counts, lists, and search queries.
* [runtime/tools/google_calendar.py](/runtime/tools/google_calendar.py): Enhanced with writing support for create, delete, and update.
* [runtime/llm/client.py](/runtime/llm/client.py): Modified to accept and attach trace IDs to request payloads sent to the LiteLLM gateway.

### Testing Suite (100% Green Build)
* [test/test_gmail_actions.py](/test/test_gmail_actions.py): Fixed patching decorator target namespaces (`tools.gmail_actions` vs `runtime.tools.gmail_actions`) and upgraded loop execution to `asyncio.run()`.
* [test/test_gmail_check.py](/test/test_gmail_check.py): Fixed mock namespaces and loop teardown exceptions.
* **Result**: **279 unit tests passed successfully** without any internal test errors or unawaited coroutines warnings.

---

## Backward Compatibility ✅
* **No Database Migrations**: `kairos.db` remains 100% compatible.
* **Session Compatibility**: Existing multi-turn history files in `data/sessions/` load cleanly.

---

## Observability Status (Langfuse Tracing)

In this version, the basic monitoring and tracing pipeline is operational:

### What's Working
* **Trace Navigation**: Full tracking for root traces (`kairos-process`) and nested child spans (`classifier`, `context-assembly`, and LiteLLM's `request-0/1/2` query calls).
* **Multi-Turn Sessions**: The Sessions page tracks active session IDs, session creation timestamps, conversation durations, and links all multi-turn conversation traces.
* **Latency Profile Analysis**: Tracks latency percentiles (P50, P90, P95, P99) for the main execution process, classifier chains, context assembly, and LLM generations.
* **Observation Logs**: Counts and registers total trace volume, observation counts, and trace/observation histories over time.

---

## Next Up (Mark 4)

* **LiteLLM Model Cost Mapping**: Configure LiteLLM pricing overrides for model aliases (`tier1`, `tier2`, `tier3`) to enable input/output cost calculation, token counts, and usage stats logging in Langfuse.
* **Telemetry Diagnostics Dashboard**: Map Time to First Token metrics and average token output speeds per model inside the dashboard analytics.
* **User Feedback & Scores**: Integrate thumbs up/down or rating selectors in the Telegram bot and WebUI to collect and trace user scores.
* **Dynamic Environments**: Support customized environment tags (e.g. `production`, `staging`) read from `.env` dynamically rather than defaulting to `default`.

