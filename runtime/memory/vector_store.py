"""
Vector store — semantic memory using plain SQLite + Python cosine similarity.

Uses nomic-embed-text via Ollama for embeddings (fully offline).
Stores vectors as JSON text in memory_embeddings table.
Cosine similarity computed in Python — fast enough for single-user scale.

Why not sqlite-vec:
    sqlite-vec requires loading a native DLL extension which is blocked
    by Windows AppLocker policies on many machines. This implementation
    has zero native dependencies — pure Python + standard sqlite3.

Performance:
    Linear scan over all stored embeddings. At 10,000 turns (~2 years
    of heavy daily use) this takes ~50ms in Python. Acceptable.
    If you ever exceed 100k embeddings, switch to sqlite-vec or Qdrant.

API is identical to the sqlite-vec version — nothing else needs to change.
"""

import asyncio
import json
import logging
import math
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

DATA_DIR           = os.getenv("DATA_DIR", "./data")
DB_PATH            = os.path.join(DATA_DIR, "kairos.db")
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBEDDING_DIM      = 768   # nomic-embed-text output dimension


# ── connection ────────────────────────────────────────────────────────────────

@contextmanager
def get_conn():
    """Plain sqlite3 connection — no extensions needed."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ── schema ────────────────────────────────────────────────────────────────────

def init_vector_store():
    """
    Create embedding tables if they don't exist.
    Safe to call every startup — uses IF NOT EXISTS.
    Must be called after init_db() since both write to kairos.db.
    """
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                content    TEXT    NOT NULL,
                source     TEXT    NOT NULL,
                session_id TEXT,
                created_at TEXT    DEFAULT CURRENT_TIMESTAMP,
                embedding  TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_mem_session
            ON memory_embeddings(session_id)
        """)
        conn.commit()
    logger.debug("Vector store ready (pure Python, nomic-embed-text)")


# ── embedding ─────────────────────────────────────────────────────────────────

async def _embed(text: str) -> list[float]:
    """
    Get embedding vector from nomic-embed-text via Ollama.
    Returns a list of 768 floats.
    Raises on network error or model not found.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


# ── cosine similarity ─────────────────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two vectors. Returns -1.0 to 1.0.
    1.0 = identical direction, 0.0 = orthogonal, -1.0 = opposite.

    Why cosine and not euclidean:
        Cosine ignores magnitude — only direction matters.
        "I love coffee" and "I really really love coffee" should
        be very similar even though the second is longer.
        Cosine handles this naturally.
    """
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Distance = 1 - similarity. 0.0 = identical, 2.0 = opposite."""
    return 1.0 - _cosine_similarity(a, b)


# ── public API ────────────────────────────────────────────────────────────────

async def embed_and_store(
    content: str,
    source: str,
    session_id: str | None = None,
) -> int:
    """
    Embed text and store with metadata. Returns the row ID.

    Args:
        content:    text to embed — usually "USER: ...\nASSISTANT: ..."
        source:     provenance label — "conversation", "note", "article"
        session_id: which session this came from

    Called by writeback.py after every completed response.
    """
    vector = await _embed(content)

    def _store():
        with get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO memory_embeddings
                   (content, source, session_id, created_at, embedding)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    content,
                    source,
                    session_id,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(vector),
                ),
            )
            conn.commit()
            return cur.lastrowid

    row_id = await asyncio.to_thread(_store)
    logger.debug("Stored embedding rowid=%d source=%s", row_id, source)
    return row_id


async def search(
    query: str,
    top_k: int = 5,
    session_id: str | None = None,
) -> list[dict]:
    """
    Find the most semantically similar stored memories for a query.

    Returns list of dicts sorted by distance ascending (most similar first).
    distance is cosine distance: 0.0 = identical, 1.0 = unrelated.
    """
    query_vec = await _embed(query)

    def _scan():
        with get_conn() as conn:
            if session_id:
                rows = conn.execute(
                    "SELECT * FROM memory_embeddings WHERE session_id = ?",
                    (session_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memory_embeddings"
                ).fetchall()

        results = []
        for row in rows:
            stored_vec = json.loads(row["embedding"])
            dist = _cosine_distance(query_vec, stored_vec)
            results.append({
                "id":         row["id"],
                "content":    row["content"],
                "source":     row["source"],
                "session_id": row["session_id"],
                "created_at": row["created_at"],
                "distance":   dist,
            })

        # Sort by distance — closest first
        results.sort(key=lambda x: x["distance"])
        return results[:top_k]

    return await asyncio.to_thread(_scan)


async def search_as_context(query: str, top_k: int = 5) -> str:
    """
    Search and format results as a context block for the prompt.
    Returns empty string if nothing relevant found.
    Called by orchestrator context assembler.
    """
    try:
        results = await search(query, top_k=top_k)
    except Exception as e:
        logger.warning("Vector search failed: %s", e)
        return ""

    if not results:
        return ""

    lines = []
    for r in results:
        # Only include reasonably similar results
        # distance < 0.3 = very similar, > 0.6 = probably noise
        if r["distance"] > 0.5:
            continue
        lines.append(
            f"[{r['source']} | {r['created_at'][:10]}]\n{r['content']}"
        )

    if not lines:
        return ""

    return "Relevant past context:\n" + "\n\n".join(lines)