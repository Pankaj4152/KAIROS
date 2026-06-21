"""
Notes tool — capture fleeting thoughts, decisions, and reference material.

This complements the existing conversation memory system (memory/vector_store.py
embeds every conversation turn automatically) by giving the user an explicit,
named, taggable note — something you deliberately write down, not something
extracted from a chat turn.

New tables (this tool owns them — nothing else in Kairos writes to these):
    notes      — id, title, body, tags, link_type, link_id, created_at, updated_at
    notes_fts  — FTS5 virtual table mirroring notes(title, body) for fast
                 keyword search, kept in sync via triggers (no manual sync code)

Two search modes:
    keyword   — SQLite FTS5 full-text search over title + body. Fast, exact-ish,
                works offline with zero extra cost.
    semantic  — delegates to memory/vector_store.py's existing embed_and_store()
                / search() functions (already async, already wired to the
                LiteLLM embeddings endpoint). Notes are embedded with
                source="note" so they're distinguishable from
                source="conversation" turns in the same memory_embeddings table.
                This reuses infrastructure instead of standing up a second
                vector store.

Linking: a note can optionally reference another entity (a task or habit) via
    link_type ('task' | 'habit') + link_id. This is informational only in
    Phase 1 — no foreign key enforcement, no cascading deletes — so linking
    a note to a task that's later deleted doesn't break anything; the note
    just keeps a now-dangling reference, same as a sticky note that outlives
    the meeting it was about.

Actions:
    create           — write a new note
    list             — list recent notes, optionally filtered by tag
    get              — fetch one note's full content by ID
    search           — keyword search (FTS5) over title + body
    semantic_search   — meaning-based search via the existing vector store
    update           — change title, body, or tags on an existing note
    delete           — permanently remove a note
    link             — attach a note to a task or habit

Env vars required: none for keyword search. semantic_search depends on the
same LITELLM_BASE_URL / EMBEDDING_MODEL config that vector_store.py already
uses — if that's not configured, semantic_search degrades to an error string
and keyword search still works.
"""

import asyncio
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DATA_DIR = os.getenv("DATA_DIR", "./data")
DB_PATH  = os.path.join(DATA_DIR, "kairos.db")

_VALID_LINK_TYPES = ("task", "habit")


# ── connection ─────────────────────────────────────────────────────────────────

@contextmanager
def _get_conn():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        conn.close()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """
    Create notes table + FTS5 index + sync triggers if they don't exist.
    Guarded with IF NOT EXISTS throughout — safe to call on every connection.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            body        TEXT NOT NULL DEFAULT '',
            tags        TEXT,
            link_type   TEXT,
            link_id     INTEGER,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_tags ON notes(tags)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_notes_link ON notes(link_type, link_id)"
    )

    # FTS5 virtual table — mirrors title + body for fast keyword search.
    # content='notes' makes this an external-content table: it stores no
    # text of its own, just the index, pointing back at notes.rowid.
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
            title, body, content='notes', content_rowid='id'
        )
    """)

    # Triggers keep notes_fts in sync automatically — no manual sync code
    # needed anywhere else in this file.
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
            INSERT INTO notes_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
            INSERT INTO notes_fts(notes_fts, rowid, title, body)
            VALUES ('delete', old.id, old.title, old.body);
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
            INSERT INTO notes_fts(notes_fts, rowid, title, body)
            VALUES ('delete', old.id, old.title, old.body);
            INSERT INTO notes_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
        END
    """)
    conn.commit()


@contextmanager
def _conn_ready():
    with _get_conn() as conn:
        _ensure_schema(conn)
        yield conn


# ── validation ─────────────────────────────────────────────────────────────────

def _validate_link_type(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().lower()
    if value not in _VALID_LINK_TYPES:
        raise ValueError(f"link_type must be one of {_VALID_LINK_TYPES} — got {value!r}")
    return value


def _normalize_tags(tags: str | list[str] | None) -> str | None:
    """Accepts a comma-separated string or a list; stores as comma-separated."""
    if tags is None:
        return None
    if isinstance(tags, list):
        parts = [t.strip().lower() for t in tags if t.strip()]
    else:
        parts = [t.strip().lower() for t in tags.split(",") if t.strip()]
    return ",".join(parts) if parts else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── formatting ─────────────────────────────────────────────────────────────────

def _snippet(body: str, length: int = 100) -> str:
    body = body.replace("\n", " ").strip()
    return body if len(body) <= length else body[:length].rstrip() + "..."


def _fmt_note_row(row: dict) -> str:
    tags = f"  tags: {row['tags']}" if row["tags"] else ""
    link = f"  → {row['link_type']}#{row['link_id']}" if row["link_type"] else ""
    return (
        f"  #{row['id']:<4} {row['created_at'][:10]}  {row['title']}\n"
        f"        {_snippet(row['body'])}{tags}{link}"
    )


# ── action handlers ───────────────────────────────────────────────────────────

def _do_create(title, body, tags, link_type, link_id) -> str:
    title = (title or "").strip()
    if not title:
        raise ValueError("title must not be empty")
    body = (body or "").strip()
    tags_str = _normalize_tags(tags)
    link_type = _validate_link_type(link_type)
    if link_type and link_id is None:
        raise ValueError("link_id is required when link_type is set")

    now = _now_iso()
    with _conn_ready() as conn:
        cur = conn.execute(
            "INSERT INTO notes (title, body, tags, link_type, link_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (title, body, tags_str, link_type, link_id, now, now),
        )
        conn.commit()
        note_id = cur.lastrowid

    tag_str = f" [tags: {tags_str}]" if tags_str else ""
    return f"Created note #{note_id}: \"{title}\"{tag_str}"


def _do_list(tag, limit) -> str:
    query  = "SELECT * FROM notes WHERE 1=1"
    params: list = []

    if tag:
        query += " AND (',' || tags || ',') LIKE ?"
        params.append(f"%,{tag.strip().lower()},%")

    query += " ORDER BY updated_at DESC LIMIT ?"
    params.append(max(1, min(limit, 100)))

    with _conn_ready() as conn:
        rows = [dict(r) for r in conn.execute(query, params).fetchall()]

    if not rows:
        return "No notes found."

    lines = [f"Notes ({len(rows)}):"]
    lines.extend(_fmt_note_row(r) for r in rows)
    return "\n".join(lines)


def _do_get(note_id) -> str:
    if note_id is None:
        raise ValueError("note_id is required")

    with _conn_ready() as conn:
        row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()

    if row is None:
        return f"Error: no note found with id {note_id}."

    row = dict(row)
    tags = f"\nTags: {row['tags']}" if row["tags"] else ""
    link = f"\nLinked to: {row['link_type']}#{row['link_id']}" if row["link_type"] else ""
    return (
        f"#{row['id']}: {row['title']}\n"
        f"Created: {row['created_at']}  Updated: {row['updated_at']}{tags}{link}\n\n"
        f"{row['body']}"
    )


def _do_search(query, limit) -> str:
    query = (query or "").strip()
    if not query:
        raise ValueError("query must not be empty")

    # FTS5 query syntax treats most punctuation specially — sanitise to a
    # simple AND-of-terms match so user text like "what's next?" doesn't
    # raise a syntax error from the FTS5 parser.
    safe_terms = "".join(c if c.isalnum() or c.isspace() else " " for c in query)
    terms = [t for t in safe_terms.split() if t]
    if not terms:
        return f"No notes found matching '{query}'."
    fts_query = " AND ".join(f'"{t}"*' for t in terms)

    with _conn_ready() as conn:
        try:
            rows = [dict(r) for r in conn.execute(
                "SELECT notes.* FROM notes "
                "JOIN notes_fts ON notes.id = notes_fts.rowid "
                "WHERE notes_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (fts_query, max(1, min(limit, 100))),
            ).fetchall()]
        except sqlite3.OperationalError as e:
            logger.warning("FTS5 query failed for %r: %s", query, e)
            return f"Error: search query could not be parsed — {e}"

    if not rows:
        return f"No notes found matching '{query}'."

    lines = [f"Search results for '{query}' ({len(rows)}):"]
    lines.extend(_fmt_note_row(r) for r in rows)
    return "\n".join(lines)


async def _do_semantic_search(query: str, top_k: int) -> str:
    """
    Delegates to memory/vector_store.py — reuses the same embedding
    pipeline and memory_embeddings table that conversation turns use,
    filtered to source='note'.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("query must not be empty")

    try:
        from memory.vector_store import search as vector_search
    except ImportError as e:
        return f"Error: semantic search unavailable — vector_store import failed: {e}"

    try:
        results = await vector_search(query, top_k=top_k)
    except Exception as e:
        logger.warning("Semantic search failed: %s", e)
        return f"Error: semantic search failed — {e}"

    note_results = [r for r in results if r.get("source") == "note"]
    if not note_results:
        return f"No semantically related notes found for '{query}'."

    lines = [f"Semantic search results for '{query}' ({len(note_results)}):"]
    for r in note_results:
        lines.append(
            f"  [{r['created_at'][:10]}] (similarity: {(1 - r['distance']) * 100:.0f}%)\n"
            f"        {_snippet(r['content'], 150)}"
        )
    return "\n".join(lines)


def _do_update(note_id, title, body, tags) -> str:
    if note_id is None:
        raise ValueError("note_id is required")

    with _conn_ready() as conn:
        existing = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        if existing is None:
            return f"Error: no note found with id {note_id}."

        updates: list[str] = []
        params: list = []
        changed: list[str] = []

        if title is not None:
            title = title.strip()
            if not title:
                raise ValueError("title must not be empty")
            updates.append("title = ?")
            params.append(title)
            changed.append("title")

        if body is not None:
            updates.append("body = ?")
            params.append(body)
            changed.append("body")

        if tags is not None:
            tags_str = _normalize_tags(tags)
            updates.append("tags = ?")
            params.append(tags_str)
            changed.append("tags")

        if not updates:
            return f"No changes specified for note #{note_id}."

        updates.append("updated_at = ?")
        params.append(_now_iso())
        params.append(note_id)

        conn.execute(f"UPDATE notes SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()

    return f"Updated note #{note_id}: " + ", ".join(changed)


def _do_delete(note_id) -> str:
    if note_id is None:
        raise ValueError("note_id is required")

    with _conn_ready() as conn:
        existing = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        if existing is None:
            return f"Error: no note found with id {note_id}."
        conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        conn.commit()

    return f"Deleted note #{note_id}: \"{existing['title']}\""


def _do_link(note_id, link_type, link_id) -> str:
    if note_id is None:
        raise ValueError("note_id is required")
    link_type = _validate_link_type(link_type)
    if link_type is None or link_id is None:
        raise ValueError("link_type and link_id are both required")

    with _conn_ready() as conn:
        existing = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        if existing is None:
            return f"Error: no note found with id {note_id}."

        conn.execute(
            "UPDATE notes SET link_type = ?, link_id = ?, updated_at = ? WHERE id = ?",
            (link_type, link_id, _now_iso(), note_id),
        )
        conn.commit()

    return f"Linked note #{note_id} (\"{existing['title']}\") to {link_type}#{link_id}."


# ── public async entrypoint ───────────────────────────────────────────────────

async def notes(
    action: str,
    note_id: int | None = None,
    title: str | None = None,
    body: str | None = None,
    tags: str | None = None,
    tag: str | None = None,
    query: str | None = None,
    link_type: str | None = None,
    link_id: int | None = None,
    limit: int = 20,
    top_k: int = 5,
) -> str:
    """
    Capture, search, and manage notes — fleeting thoughts, decisions, and
    reference material the user explicitly wants to save and recall later.

    Args:
        action:     which operation to perform (see below). Required.
        note_id:    target note ID. Required for get/update/delete/link.
        title:      note title. Required for create; optional for update.
        body:       note content. Optional for create (defaults to empty);
                    optional for update.
        tags:       comma-separated tags for create/update, e.g. "ml,research".
        tag:        single tag to filter by for action='list'.
        query:      search text for action='search' or 'semantic_search'.
        link_type:  'task' or 'habit', for action='create' or 'link'.
        link_id:    the ID of the linked task/habit, for action='create' or 'link'.
        limit:      max rows for list/search. Default 20, max 100.
        top_k:      max results for semantic_search. Default 5.

    Actions:
        create           — write a new note. Requires: title. Optional: body, tags,
                           link_type+link_id.
        list             — list recent notes, newest-updated first. Optional tag filter.
        get              — fetch one note's full content. Requires: note_id.
        search           — fast keyword search (FTS5) over title + body. Requires: query.
        semantic_search  — meaning-based search via the existing memory vector store
                           (finds related notes even without exact keyword overlap).
                           Requires: query.
        update           — change title, body, or tags. Requires: note_id + at least
                           one field.
        delete           — permanently remove a note. Requires: note_id.
        link             — attach a note to a task or habit. Requires: note_id,
                           link_type, link_id.

    Returns a plain string in all cases. Never raises to the caller.
    """
    action = (action or "").strip().lower()
    valid_actions = {
        "create", "list", "get", "search", "semantic_search",
        "update", "delete", "link",
    }
    if action not in valid_actions:
        return f"Error: Unknown action '{action}'. Valid: {', '.join(sorted(valid_actions))}."

    try:
        if action == "create":
            return await asyncio.to_thread(_do_create, title, body, tags, link_type, link_id)

        if action == "list":
            return await asyncio.to_thread(_do_list, tag, limit)

        if action == "get":
            return await asyncio.to_thread(_do_get, note_id)

        if action == "search":
            return await asyncio.to_thread(_do_search, query, limit)

        if action == "semantic_search":
            return await _do_semantic_search(query, top_k)

        if action == "update":
            return await asyncio.to_thread(_do_update, note_id, title, body, tags)

        if action == "delete":
            return await asyncio.to_thread(_do_delete, note_id)

        if action == "link":
            return await asyncio.to_thread(_do_link, note_id, link_type, link_id)

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.exception("notes() failed for action=%r", action)
        return f"Error: Notes tool failed unexpectedly — {type(e).__name__}: {e}"

    return "Error: Unhandled action path."