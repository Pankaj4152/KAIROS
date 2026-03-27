"""
Tests for runtime/memory/vector_store.py

Covers:
  - Cosine similarity: identical, orthogonal, opposite, zero vectors
  - Cosine distance = 1 - similarity
  - Schema init creates memory_embeddings table
  - search_as_context filters high-distance results
"""

import logging
import math
import os

import pytest

logger = logging.getLogger(__name__)


# ── cosine similarity (pure math, no I/O) ─────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        from memory.vector_store import _cosine_similarity
        a = [1.0, 2.0, 3.0]
        result = _cosine_similarity(a, a)
        logger.info("Identical vectors → similarity=%.4f (expected ~1.0)", result)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        from memory.vector_store import _cosine_similarity
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        result = _cosine_similarity(a, b)
        logger.info("Orthogonal vectors → similarity=%.4f (expected 0.0)", result)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        from memory.vector_store import _cosine_similarity
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        result = _cosine_similarity(a, b)
        logger.info("Opposite vectors → similarity=%.4f (expected -1.0)", result)
        assert result == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        from memory.vector_store import _cosine_similarity
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        result = _cosine_similarity(a, b)
        logger.info("Zero vector → similarity=%.4f (expected 0.0)", result)
        assert result == 0.0

    def test_magnitude_invariance(self):
        """Cosine similarity should not depend on vector magnitude."""
        from memory.vector_store import _cosine_similarity
        a = [1.0, 2.0, 3.0]
        b = [10.0, 20.0, 30.0]  # same direction, 10x magnitude
        result = _cosine_similarity(a, b)
        logger.info("Same direction, different magnitude → similarity=%.4f (expected ~1.0)", result)
        assert result == pytest.approx(1.0, abs=1e-6)


class TestCosineDistance:
    def test_identical_is_zero(self):
        from memory.vector_store import _cosine_distance
        a = [1.0, 2.0, 3.0]
        result = _cosine_distance(a, a)
        logger.info("Identical vectors → distance=%.4f (expected 0.0)", result)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_opposite_is_two(self):
        from memory.vector_store import _cosine_distance
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        result = _cosine_distance(a, b)
        logger.info("Opposite vectors → distance=%.4f (expected 2.0)", result)
        assert result == pytest.approx(2.0, abs=1e-6)


# ── schema init ───────────────────────────────────────────────────────────────

class TestVectorStoreInit:
    def test_creates_embeddings_table(self, tmp_data_dir, monkeypatch):
        import sqlite3
        db_path = os.path.join(tmp_data_dir, "kairos.db")
        monkeypatch.setattr("memory.vector_store.DATA_DIR", tmp_data_dir)
        monkeypatch.setattr("memory.vector_store.DB_PATH", db_path)

        from memory.vector_store import init_vector_store
        init_vector_store()

        conn = sqlite3.connect(db_path)
        tables = [
            r[0] for r in
            conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]
        conn.close()

        logger.info("Tables after init_vector_store: %s", tables)
        assert "memory_embeddings" in tables


# ── search_as_context (mocked embedding) ──────────────────────────────────────

class TestSearchAsContext:
    def test_empty_results_returns_empty_string(self, monkeypatch):
        """When no embeddings exist, search_as_context returns ''."""
        import asyncio
        from unittest.mock import AsyncMock

        monkeypatch.setattr("memory.vector_store.search", AsyncMock(return_value=[]))

        from memory.vector_store import search_as_context
        result = asyncio.get_event_loop().run_until_complete(
            search_as_context("test query")
        )
        logger.info("Empty search result: %r", result)
        assert result == ""

    def test_filters_high_distance_results(self, monkeypatch):
        """Results with distance > 0.5 are excluded."""
        import asyncio
        from unittest.mock import AsyncMock

        mock_results = [
            {"content": "relevant", "source": "conv", "created_at": "2026-01-01T00:00:00", "distance": 0.2},
            {"content": "noise", "source": "conv", "created_at": "2026-01-01T00:00:00", "distance": 0.8},
        ]
        monkeypatch.setattr("memory.vector_store.search", AsyncMock(return_value=mock_results))

        from memory.vector_store import search_as_context
        result = asyncio.get_event_loop().run_until_complete(
            search_as_context("test query")
        )
        logger.info("Filtered result: %s", result[:100])
        assert "relevant" in result
        assert "noise" not in result
