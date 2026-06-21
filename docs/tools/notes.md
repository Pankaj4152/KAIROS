# notes

**← [Back to Tool Guide](../TOOL_GUIDE.md)**

**File:** `runtime/tools/notes.py`
**Env vars required:** none for keyword search. `semantic_search` reuses the same `LITELLM_BASE_URL`/`EMBEDDING_MODEL` config that `memory/vector_store.py` already depends on — if that's not configured, `semantic_search` degrades to an error string while keyword search keeps working
**Execution pattern:** agentic
**Added:** Mark 3

Explicit, named, taggable notes — fleeting thoughts, decisions, reference material the user deliberately writes down. This is distinct from the automatic conversation-memory system (`memory/vector_store.py`, which embeds every conversation turn): notes are intentional, titled artifacts, not extracted chat history.

Owns two SQLite objects, created on first use: a `notes` table and a `notes_fts` FTS5 virtual table (kept in sync via triggers — no manual sync code needed anywhere else).

## When it's used

- User explicitly says "note this," "save this," "remember this as a note," wants to search past notes, or wants to attach a note to a task/habit
- Classifier sets `domains: [notes]` and `tools_needed: [notes]`

## Parameters

| Parameter     | Type    | Required for                                    | Description |
|----------------|---------|-----------------------------------------------------|--------------|
| `action`       | string  | all                                                   | Which operation to perform |
| `note_id`      | integer | `get`, `update`, `delete`, `link`                       | Target note ID |
| `title`        | string  | `create`; optional for `update`                          | Note title |
| `body`         | string  | optional, `create` (defaults `""`)/`update`                | Note content |
| `tags`         | string  | optional, `create`/`update`                                  | Comma-separated, e.g. `"ml,research"` |
| `tag`          | string  | optional, `list` filter                                        | Single tag to filter by |
| `query`        | string  | `search`/`semantic_search`                                        | Search text |
| `link_type`    | string  | `create`/`link`                                                      | `"task"` or `"habit"` |
| `link_id`      | integer | `create`/`link` (required together with `link_type`)                   | ID of the task/habit to link to |
| `limit`        | integer | optional, `list`/`search`                                                | Default 20, max 100 |
| `top_k`        | integer | optional, `semantic_search`                                                | Default 5 |

## Actions

| Action              | What it does | Required params |
|----------------------|---------------|--------------------|
| `create`             | Write a new note. Optionally link to a task/habit at creation time | `title` |
| `list`               | Recent notes, newest-updated first, optional single-tag filter | — |
| `get`                  | Full content of one note | `note_id` |
| `search`                 | Fast keyword search (SQLite FTS5) over title + body. Punctuation in the query is sanitized, not rejected | `query` |
| `semantic_search`          | Meaning-based search — delegates to `memory/vector_store.py`'s existing embedding pipeline, filtered to `source='note'` so conversation turns never leak into note results | `query` |
| `update`                      | Change `title`, `body`, and/or `tags` | `note_id` + ≥1 field |
| `delete`                        | Permanently remove a note (FTS index entry cleaned up automatically via trigger) | `note_id` |
| `link`                             | Attach an existing note to a task or habit | `note_id`, `link_type`, `link_id` |

## Example LLM calls

**Create with tags:**
```json
{ "name": "notes", "input": { "action": "create", "title": "Caching strategy", "body": "Use Redis for hot data", "tags": "kairos,architecture" } }
```

**Keyword search:**
```json
{ "name": "notes", "input": { "action": "search", "query": "caching" } }
```

**Semantic search (finds related notes without exact keyword overlap):**
```json
{ "name": "notes", "input": { "action": "semantic_search", "query": "ideas about speeding up reads" } }
```

**Link an existing note to a task:**
```json
{ "name": "notes", "input": { "action": "link", "note_id": 7, "link_type": "task", "link_id": 12 } }
```

## Return format (search)

```
Search results for 'caching' (2):
  #3    2026-06-15  Caching strategy
        Use Redis for hot data  tags: kairos,architecture
  #5    2026-06-16  Meeting recap
        Discussed caching layer design
```

## Return format (get)

```
#3: Caching strategy
Created: 2026-06-15T09:00:00+00:00  Updated: 2026-06-15T09:00:00+00:00
Tags: kairos,architecture
Linked to: task#12

Use Redis for hot data
```

## Failure modes

| Condition                                  | Return value |
|-----------------------------------------------|-----------------|
| Unknown action                                  | `Error: Unknown action '<action>'. Valid: create, delete, get, link, list, search, semantic_search, update.` |
| `create` with empty `title`                       | `Error: title must not be empty` |
| `create`/`link` with `link_type` but no `link_id`   | `Error: link_id is required when link_type is set` |
| Invalid `link_type`                                   | `Error: link_type must be one of ('task', 'habit') — got '<value>'` |
| `get`/`update`/`delete`/`link` on nonexistent `note_id` | `Error: no note found with id <id>.` |
| `update` called with no fields                            | `No changes specified for note #<id>.` |
| `search`/`semantic_search` with empty query                  | `Error: query must not be empty` |
| FTS5 query fails to parse (rare — punctuation is pre-sanitized) | `Error: search query could not be parsed — <detail>` |
| `semantic_search` — vector store import unavailable               | `Error: semantic search unavailable — vector_store import failed: <detail>` |
| `semantic_search` — embedding endpoint fails                         | `Error: semantic search failed — <detail>` |
| `semantic_search` — no note-source matches                              | `No semantically related notes found for '<query>'.` |
| Unexpected exception                                                       | `Error: Notes tool failed unexpectedly — <ExceptionType>: <detail>` |

## Notes

- Linking is informational only in Phase 1 — no foreign key enforcement, no cascading deletes. If you link a note to a task that's later deleted, the note keeps a dangling reference rather than breaking, the same way a sticky note outlives the meeting it was about.
- `semantic_search` results are tagged `source='note'` in the shared `memory_embeddings` table specifically so they're distinguishable from `source='conversation'` turns — the same table powers both, but results never cross-contaminate.
- Use `search` (fast, free, exact-ish) by default; reach for `semantic_search` when the user describes a note by meaning rather than likely keywords.
