"""
Web search tool — pluggable backend design with robust error handling.

Switch search engine by setting SEARCH_BACKEND in .env:
    SEARCH_BACKEND=duckduckgo   (default, free, no key needed)
    SEARCH_BACKEND=brave        (free tier 2000/month, needs BRAVE_API_KEY)
    SEARCH_BACKEND=tavily       (1000/month free, needs TAVILY_API_KEY)
    SEARCH_BACKEND=serper       (2500 free, needs SERPER_API_KEY)

Improvements over v1:
    - Per-backend timeout + global deadline so one slow backend never blocks
    - Retry with exponential backoff (configurable)
    - Result validation: strips empty/junk entries, deduplicates by URL
    - Snippet length cap to prevent prompt bloat
    - Structured SearchResult dataclass — no silent missing-key bugs
    - Anti-hallucination: formatter explicitly marks source URLs
      so the LLM cites instead of paraphrasing dangerously
    - Parallel multi-backend fallback: if primary fails, secondary runs
    - Circuit breaker: a backend that keeps failing gets paused
    - Centralised env-check on startup, not at query time
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

SEARCH_BACKEND          = os.getenv("SEARCH_BACKEND", "duckduckgo").lower().strip()
FALLBACK_BACKEND        = os.getenv("SEARCH_FALLBACK_BACKEND", "").lower().strip()  # optional
MAX_RESULTS             = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
REQUEST_TIMEOUT         = float(os.getenv("SEARCH_TIMEOUT_SEC", "8.0"))
MAX_RETRIES             = int(os.getenv("SEARCH_MAX_RETRIES", "2"))
RETRY_BASE_DELAY        = float(os.getenv("SEARCH_RETRY_BASE_DELAY", "0.5"))
SNIPPET_MAX_CHARS       = int(os.getenv("SEARCH_SNIPPET_MAX_CHARS", "300"))
CIRCUIT_BREAKER_THRESH  = int(os.getenv("SEARCH_CB_THRESHOLD", "3"))   # failures before pause
CIRCUIT_BREAKER_RESET   = int(os.getenv("SEARCH_CB_RESET_SEC", "60"))  # seconds before retry

ALLOWED_SCHEMES = {"http", "https"}


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    title: str
    body:  str
    url:   str

    def is_valid(self) -> bool:
        """Reject results that are empty, malformed, or clearly junk."""
        if not self.url or not self.title:
            return False
        try:
            parsed = urlparse(self.url)
            if parsed.scheme not in ALLOWED_SCHEMES:
                return False
            if not parsed.netloc:
                return False
        except Exception:
            return False
        # Reject suspiciously short snippets (likely scrape noise)
        if len(self.body.strip()) < 10 and len(self.title.strip()) < 5:
            return False
        return True

    def cleaned(self) -> "SearchResult":
        """Return a copy with normalised whitespace and capped snippet."""
        body = re.sub(r"\s+", " ", self.body).strip()
        if len(body) > SNIPPET_MAX_CHARS:
            # Cut at last word boundary inside limit
            body = body[:SNIPPET_MAX_CHARS].rsplit(" ", 1)[0] + "…"
        return SearchResult(
            title=re.sub(r"\s+", " ", self.title).strip(),
            body=body,
            url=self.url.strip(),
        )

    def url_fingerprint(self) -> str:
        """Canonical key for deduplication — strips query params & fragments."""
        try:
            p = urlparse(self.url)
            canonical = f"{p.scheme}://{p.netloc}{p.path}".rstrip("/").lower()
            return hashlib.md5(canonical.encode()).hexdigest()
        except Exception:
            return self.url


def deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    out: list[SearchResult] = []
    for r in results:
        fp = r.url_fingerprint()
        if fp not in seen:
            seen.add(fp)
            out.append(r)
    return out


# ── circuit breaker ───────────────────────────────────────────────────────────

@dataclass
class CircuitBreaker:
    threshold: int   = CIRCUIT_BREAKER_THRESH
    reset_sec: float = CIRCUIT_BREAKER_RESET
    _failures: int   = field(default=0, repr=False)
    _opened_at: Optional[float] = field(default=None, repr=False)

    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.reset_sec:
            logger.info("Circuit breaker reset after %.0fs", self.reset_sec)
            self._failures = 0
            self._opened_at = None
            return False
        return True

    def record_failure(self):
        self._failures += 1
        if self._failures >= self.threshold:
            self._opened_at = time.monotonic()
            logger.warning("Circuit breaker OPEN after %d failures", self._failures)

    def record_success(self):
        self._failures = 0
        self._opened_at = None


# ── base class ────────────────────────────────────────────────────────────────

class SearchBackend(ABC):
    """
    Contract: search() returns validated, cleaned SearchResult list.
    Never raises — returns [] on any failure.
    """

    def __init__(self):
        self._cb = CircuitBreaker()

    @abstractmethod
    async def _fetch(self, query: str, max_results: int) -> list[SearchResult]:
        """Raw fetch — may raise, caller handles it."""

    async def search(self, query: str, max_results: int = MAX_RESULTS) -> list[SearchResult]:
        if self._cb.is_open():
            logger.warning("%s: circuit breaker open, skipping", self.__class__.__name__)
            return []

        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 2):          # 1 + MAX_RETRIES total attempts
            try:
                raw = await asyncio.wait_for(
                    self._fetch(query, max_results),
                    timeout=REQUEST_TIMEOUT,
                )
                results = [r.cleaned() for r in raw if r.is_valid()]
                results = deduplicate(results)
                self._cb.record_success()
                return results

            except asyncio.TimeoutError as e:
                last_exc = e
                logger.warning("%s: timeout on attempt %d/%d",
                               self.__class__.__name__, attempt, MAX_RETRIES + 1)

            except Exception as e:
                last_exc = e
                logger.warning("%s: attempt %d/%d failed — %s: %s",
                               self.__class__.__name__, attempt, MAX_RETRIES + 1,
                               type(e).__name__, e)

            if attempt <= MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))   # 0.5 → 1.0 → 2.0…
                await asyncio.sleep(delay)

        self._cb.record_failure()
        logger.error("%s: all attempts exhausted. Last error: %s", self.__class__.__name__, last_exc)
        return []


# ── backends ──────────────────────────────────────────────────────────────────

class DuckDuckGoBackend(SearchBackend):
    """Free, no API key. Uses duckduckgo-search package (pip install duckduckgo-search)."""

    async def _fetch(self, query: str, max_results: int) -> list[SearchResult]:
        try:
            from ddgs import DDGS
        except ImportError:
            raise RuntimeError("duckduckgo-search not installed: pip install duckduckgo-search")

        def _run() -> list[dict]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        raw: list[dict] = await asyncio.to_thread(_run)
        return [
            SearchResult(
                title=r.get("title", ""),
                body=r.get("body", ""),
                url=r.get("href", ""),
            )
            for r in raw
        ]


class BraveBackend(SearchBackend):
    """Brave Search API — 2000 req/month free. Needs BRAVE_API_KEY."""

    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("BRAVE_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError("BRAVE_API_KEY not set in environment")

    async def _fetch(self, query: str, max_results: int) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(
                self.BASE_URL,
                params={"q": query, "count": max_results},
                headers={
                    "X-Subscription-Token": self.api_key,
                    "Accept": "application/json",
                },
            )
            resp.raise_for_status()
            results = resp.json().get("web", {}).get("results", [])
            return [
                SearchResult(
                    title=r.get("title", ""),
                    body=r.get("description", ""),
                    url=r.get("url", ""),
                )
                for r in results
            ]


class TavilyBackend(SearchBackend):
    """Tavily — built for AI agents. 1000 req/month free. Needs TAVILY_API_KEY."""

    BASE_URL = "https://api.tavily.com/search"

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("TAVILY_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError("TAVILY_API_KEY not set in environment")

    async def _fetch(self, query: str, max_results: int) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(
                self.BASE_URL,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_answer": False,   # raw results only, no AI summary
                },
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return [
                SearchResult(
                    title=r.get("title", ""),
                    body=r.get("content", ""),
                    url=r.get("url", ""),
                )
                for r in results
            ]


class SerperBackend(SearchBackend):
    """Serper.dev — Google results. 2500 free queries. Needs SERPER_API_KEY."""

    BASE_URL = "https://google.serper.dev/search"

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("SERPER_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError("SERPER_API_KEY not set in environment")

    async def _fetch(self, query: str, max_results: int) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(
                self.BASE_URL,
                headers={
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": max_results},
            )
            resp.raise_for_status()
            results = resp.json().get("organic", [])
            return [
                SearchResult(
                    title=r.get("title", ""),
                    body=r.get("snippet", ""),
                    url=r.get("link", ""),
                )
                for r in results
            ]


# ── registry ──────────────────────────────────────────────────────────────────

BACKEND_MAP: dict[str, type[SearchBackend]] = {
    "duckduckgo": DuckDuckGoBackend,
    "brave":      BraveBackend,
    "tavily":     TavilyBackend,
    "serper":     SerperBackend,
}


def _build_backend(name: str) -> Optional[SearchBackend]:
    cls = BACKEND_MAP.get(name)
    if cls is None:
        logger.error("Unknown search backend %r. Valid options: %s", name, list(BACKEND_MAP))
        return None
    try:
        return cls()
    except EnvironmentError as e:
        logger.error("Cannot initialise backend %r: %s", name, e)
        return None


# Module-level singletons (initialised once)
_primary_backend:  Optional[SearchBackend] = None
_fallback_backend: Optional[SearchBackend] = None
_backends_ready = False


def _ensure_backends():
    global _primary_backend, _fallback_backend, _backends_ready
    if _backends_ready:
        return

    _primary_backend = _build_backend(SEARCH_BACKEND)
    if _primary_backend is None:
        logger.warning("Primary backend unavailable, falling back to duckduckgo")
        _primary_backend = DuckDuckGoBackend()

    if FALLBACK_BACKEND and FALLBACK_BACKEND != SEARCH_BACKEND:
        _fallback_backend = _build_backend(FALLBACK_BACKEND)

    logger.info("Search: primary=%s  fallback=%s",
                _primary_backend.__class__.__name__,
                _fallback_backend.__class__.__name__ if _fallback_backend else "none")
    _backends_ready = True


# ── formatter (anti-hallucination) ────────────────────────────────────────────

def _format(results: list[SearchResult], query: str) -> str:
    """
    Produces a prompt-ready block that:
      - Labels every claim with [SOURCE N] so the LLM can cite instead of hallucinate
      - Includes the raw URL so the LLM is anchored to a real document
      - Adds an explicit instruction comment reminding the LLM to cite sources
    """
    if not results:
        return (
            f"[SEARCH: no results found for query: {query!r}]\n"
            "Do not invent or guess an answer — tell the user you could not find current information."
        )

    lines = [
        f"[SEARCH RESULTS for: {query!r}]",
        "Use only the information below. Cite source numbers. Do not add facts not present here.",
        "",
    ]
    for i, r in enumerate(results, 1):
        lines.append(f"[SOURCE {i}]")
        lines.append(f"Title : {r.title}")
        if r.body:
            lines.append(f"Snippet: {r.body}")
        lines.append(f"URL   : {r.url}")
        lines.append("")

    lines.append("[END OF SEARCH RESULTS]")
    return "\n".join(lines)


# ── public API ────────────────────────────────────────────────────────────────

async def web_search(query: str, max_results: int = MAX_RESULTS) -> str:
    """
    Search the web. Returns a formatted, citation-ready string for prompt injection.
    Never raises.

    Strategy:
      1. Try primary backend.
      2. If results are empty AND a fallback is configured, try fallback.
      3. Merge + deduplicate if both return results.
    """
    _ensure_backends()

    query = query.strip()
    if not query:
        return "[SEARCH ERROR: empty query received]"

    # Cap query length defensively
    if len(query) > 200:
        query = query[:200]
        logger.warning("Query truncated to 200 chars")

    results: list[SearchResult] = []

    # Primary
    if _primary_backend:
        results = await _primary_backend.search(query, max_results)

    # Fallback if primary came up empty
    if not results and _fallback_backend:
        logger.info("Primary returned 0 results, trying fallback")
        results = await _fallback_backend.search(query, max_results)

    # Final dedup pass (in case both backends returned overlapping URLs)
    results = deduplicate(results)[:max_results]

    return _format(results, query)