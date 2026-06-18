"""
Tests for web_search.py and tool_registry.py
Run: pytest tests.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ── web_search tests ──────────────────────────────────────────────────────────

from runtime.tools.web_search import (
    SearchResult, deduplicate, _format,
    DuckDuckGoBackend, CircuitBreaker,
)


class TestSearchResult:
    def test_valid_result(self):
        r = SearchResult("Title", "Some body text here", "https://example.com")
        assert r.is_valid()

    def test_rejects_empty_url(self):
        assert not SearchResult("Title", "body", "").is_valid()

    def test_rejects_non_http_scheme(self):
        assert not SearchResult("Title", "body", "ftp://example.com").is_valid()

    def test_rejects_junk_result(self):
        assert not SearchResult("T", "x", "https://ok.com").is_valid()  # both too short

    def test_snippet_capped(self):
        long_body = "word " * 200
        r = SearchResult("Title", long_body, "https://example.com").cleaned()
        assert len(r.body) <= 305  # SNIPPET_MAX_CHARS + ellipsis

    def test_whitespace_normalised(self):
        r = SearchResult("Ti  tle\n", "body\t\tbody", "https://example.com").cleaned()
        assert "\n" not in r.title
        assert "\t" not in r.body


class TestDeduplicate:
    def test_removes_duplicate_urls(self):
        results = [
            SearchResult("A", "body", "https://example.com/page"),
            SearchResult("B", "body", "https://example.com/page/"),   # trailing slash variant
            SearchResult("C", "body", "https://other.com"),
        ]
        out = deduplicate(results)
        assert len(out) == 2

    def test_preserves_order(self):
        results = [
            SearchResult("A", "body", "https://a.com"),
            SearchResult("B", "body", "https://b.com"),
        ]
        out = deduplicate(results)
        assert out[0].url == "https://a.com"


class TestFormatter:
    def test_empty_returns_no_hallucinate_message(self):
        msg = _format([], "test query")
        assert "Do not invent" in msg

    def test_sources_numbered(self):
        results = [
            SearchResult("Title1", "body1", "https://a.com"),
            SearchResult("Title2", "body2", "https://b.com"),
        ]
        msg = _format(results, "test")
        assert "[SOURCE 1]" in msg
        assert "[SOURCE 2]" in msg
        assert "https://a.com" in msg

    def test_cite_instruction_present(self):
        results = [SearchResult("Title", "body", "https://a.com")]
        msg = _format(results, "test")
        assert "Cite source numbers" in msg


class TestCircuitBreaker:
    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(threshold=2, reset_sec=999)
        cb.record_failure()
        assert not cb.is_open()
        cb.record_failure()
        assert cb.is_open()

    def test_resets_after_time(self):
        import time
        cb = CircuitBreaker(threshold=1, reset_sec=0.01)
        cb.record_failure()
        assert cb.is_open()
        time.sleep(0.02)
        assert not cb.is_open()  # should auto-reset

    def test_success_clears_failures(self):
        cb = CircuitBreaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb._failures == 0


class TestDuckDuckGoBackend:
    @pytest.mark.anyio
    async def test_returns_empty_on_timeout(self):
        backend = DuckDuckGoBackend()
        with patch.object(backend, "_fetch", new=AsyncMock(side_effect=asyncio.TimeoutError)):
            results = await backend.search("test", 3)
        assert results == []

    @pytest.mark.anyio
    async def test_filters_invalid_results(self):
        backend = DuckDuckGoBackend()
        mock_results = [
            SearchResult("", "", ""),                          # invalid
            SearchResult("Good Title", "good body text here", "https://ok.com"),  # valid
        ]
        with patch.object(backend, "_fetch", new=AsyncMock(return_value=mock_results)):
            results = await backend.search("test", 5)
        assert len(results) == 1
        assert results[0].url == "https://ok.com"


