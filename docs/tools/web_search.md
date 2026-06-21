# web_search

**← [Back to Tool Guide](../TOOL_GUIDE.md)**

**File:** `runtime/tools/web_search.py`
**Env vars required:** none for the default backend (DuckDuckGo). Optional: `BRAVE_API_KEY`, `TAVILY_API_KEY`, `SERPER_API_KEY`
**Backend:** set via `SEARCH_BACKEND` in `.env` — `duckduckgo` (default) | `brave` | `tavily` | `serper`
**Execution pattern:** **pre-LLM** — this is the only tool in `PRE_LLM_TOOLS`. The orchestrator runs it before the first LLM call and injects results into the system prompt as context; the LLM does not call it mid-conversation.

## When it's used

- The classifier sets `intent: search` or `needs_external_tools: true`
- The user asks about rece# web_search

**← [Back to Tool Guide](../TOOL_GUIDE.md)**

**File:** `runtime/tools/web_search.py`
**Env vars required:** none for the default backend (DuckDuckGo). Optional: `BRAVE_API_KEY`, `TAVILY_API_KEY`, `SERPER_API_KEY`
**Backend:** set via `SEARCH_BACKEND` in `.env` — `duckduckgo` (default) | `brave` | `tavily` | `serper`
**Execution pattern:** **pre-LLM** — this is the only tool in `PRE_LLM_TOOLS`. The orchestrator runs it before the first LLM call and injects results into the system prompt as context; the LLM does not call it mid-conversation.

## When it's used

- The classifier sets `intent: search` or `needs_external_tools: true`
- The user asks about recent events, news, prices, or anything that may have changed since the model's training
- A factual question the model can't confidently answer from memory

## Parameters

| Parameter     | Type    | Required | Constraints   | Description                                  |
|---------------|---------|----------|---------------|-----------------------------------------------|
| `query`       | string  | yes      | 1–200 chars   | The search query                              |
| `max_results` | integer | no       | 1–10, default 5 | Number of results to return                |

## Return format

Results are formatted as a citation-ready block, not free prose — this is deliberate anti-hallucination design (`_format()` in `web_search.py`):

```
[SEARCH RESULTS for: 'India vs Australia Test series 2026 results']
Use only the information below. Cite source numbers. Do not add facts not present here.

[SOURCE 1]
Title : India win third Test to take series lead
Snippet: Australia's batting collapsed on day four as India claimed victory by an innings.
URL   : https://example-cricket-site.com/...

[SOURCE 2]
...

[END OF SEARCH RESULTS]
```

If nothing is found:

```
[SEARCH: no results found for query: 'query text here']
Do not invent or guess an answer — tell the user you could not find current information.
```

## Resilience features

- **Retry with exponential backoff** — `SEARCH_MAX_RETRIES` (default 2), base delay `SEARCH_RETRY_BASE_DELAY` (default 0.5s)
- **Per-backend circuit breaker** — opens after `SEARCH_CB_THRESHOLD` (default 3) consecutive failures, resets after `SEARCH_CB_RESET_SEC` (default 60s)
- **Optional fallback backend** — set `SEARCH_FALLBACK_BACKEND`; only tried if the primary returns zero results
- **Result hygiene** — dedup by canonical URL (strips query params/fragments), snippet capped to `SEARCH_SNIPPET_MAX_CHARS` (default 300, cut at word boundary), rejects junk results with empty/invalid URLs or near-empty title+body

## Failure modes

| Condition                          | Return value                                                            |
|-------------------------------------|---------------------------------------------------------------------------|
| Empty query                         | `[SEARCH ERROR: empty query received]`                                  |
| No results from any backend         | `[SEARCH: no results found for query: '...']` + anti-hallucination note |
| Primary backend down, no fallback set | Falls through to empty results message above                          |
| `SEARCH_BACKEND` set to an unknown value | Falls back to DuckDuckGo automatically (logged as an error)        |
| Backend missing required API key (`brave`/`tavily`/`serper`) | Falls back to DuckDuckGo                              |
| Query longer than 200 chars         | Silently truncated to 200 chars before searching                        |

## Config

```env
SEARCH_BACKEND=duckduckgo          # no key needed
SEARCH_BACKEND=brave               # needs BRAVE_API_KEY
SEARCH_BACKEND=tavily              # needs TAVILY_API_KEY
SEARCH_BACKEND=serper              # needs SERPER_API_KEY
SEARCH_FALLBACK_BACKEND=           # optional secondary backend
SEARCH_MAX_RESULTS=5
SEARCH_TIMEOUT_SEC=8.0
SEARCH_MAX_RETRIES=2
SEARCH_RETRY_BASE_DELAY=0.5
SEARCH_SNIPPET_MAX_CHARS=300
SEARCH_CB_THRESHOLD=3
SEARCH_CB_RESET_SEC=60
```nt events, news, prices, or anything that may have changed since the model's training
- A factual question the model can't confidently answer from memory

## Parameters

| Parameter     | Type    | Required | Constraints   | Description                                  |
|---------------|---------|----------|---------------|-----------------------------------------------|
| `query`       | string  | yes      | 1–200 chars   | The search query                              |
| `max_results` | integer | no       | 1–10, default 5 | Number of results to return                |

## Return format

Results are formatted as a citation-ready block, not free prose — this is deliberate anti-hallucination design (`_format()` in `web_search.py`):

```
[SEARCH RESULTS for: 'India vs Australia Test series 2026 results']
Use only the information below. Cite source numbers. Do not add facts not present here.

[SOURCE 1]
Title : India win third Test to take series lead
Snippet: Australia's batting collapsed on day four as India claimed victory by an innings.
URL   : https://example-cricket-site.com/...

[SOURCE 2]
...

[END OF SEARCH RESULTS]
```

If nothing is found:

```
[SEARCH: no results found for query: 'query text here']
Do not invent or guess an answer — tell the user you could not find current information.
```

## Resilience features

- **Retry with exponential backoff** — `SEARCH_MAX_RETRIES` (default 2), base delay `SEARCH_RETRY_BASE_DELAY` (default 0.5s)
- **Per-backend circuit breaker** — opens after `SEARCH_CB_THRESHOLD` (default 3) consecutive failures, resets after `SEARCH_CB_RESET_SEC` (default 60s)
- **Optional fallback backend** — set `SEARCH_FALLBACK_BACKEND`; only tried if the primary returns zero results
- **Result hygiene** — dedup by canonical URL (strips query params/fragments), snippet capped to `SEARCH_SNIPPET_MAX_CHARS` (default 300, cut at word boundary), rejects junk results with empty/invalid URLs or near-empty title+body

## Failure modes

| Condition                          | Return value                                                            |
|-------------------------------------|---------------------------------------------------------------------------|
| Empty query                         | `[SEARCH ERROR: empty query received]`                                  |
| No results from any backend         | `[SEARCH: no results found for query: '...']` + anti-hallucination note |
| Primary backend down, no fallback set | Falls through to empty results message above                          |
| `SEARCH_BACKEND` set to an unknown value | Falls back to DuckDuckGo automatically (logged as an error)        |
| Backend missing required API key (`brave`/`tavily`/`serper`) | Falls back to DuckDuckGo                              |
| Query longer than 200 chars         | Silently truncated to 200 chars before searching                        |

## Config

```env
SEARCH_BACKEND=duckduckgo          # no key needed
SEARCH_BACKEND=brave               # needs BRAVE_API_KEY
SEARCH_BACKEND=tavily              # needs TAVILY_API_KEY
SEARCH_BACKEND=serper              # needs SERPER_API_KEY
SEARCH_FALLBACK_BACKEND=           # optional secondary backend
SEARCH_MAX_RESULTS=5
SEARCH_TIMEOUT_SEC=8.0
SEARCH_MAX_RETRIES=2
SEARCH_RETRY_BASE_DELAY=0.5
SEARCH_SNIPPET_MAX_CHARS=300
SEARCH_CB_THRESHOLD=3
SEARCH_CB_RESET_SEC=60
```
