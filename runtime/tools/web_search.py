"""
Web search tool — pluggable backend design.

Switch search engine by setting SEARCH_BACKEND in .env:
    SEARCH_BACKEND=duckduckgo   (default, free, no key needed)
    SEARCH_BACKEND=brave        (free tier 2000/month, needs BRAVE_API_KEY)
    SEARCH_BACKEND=tavily       (1000/month free, needs TAVILY_API_KEY)
    SEARCH_BACKEND=serper       (2500 free, needs SERPER_API_KEY)

Adding a new backend:
    1. Subclass SearchBackend
    2. Implement async def search(query, max_results) -> list[dict]
       Each dict must have: title, body, url
    3. Register in _get_backend()
    That's it. Nothing else changes.
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod

import httpx

logger = logging.getLogger(__name__)

SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "duckduckgo")
MAX_RESULTS    = int(os.getenv("SEARCH_MAX_RESULTS", "5"))


# ── base class ────────────────────────────────────────────────────────────────

class SearchBackend(ABC):
    """
    All backends return the same shape — list of dicts with title, body, url.
    The formatter below turns that into a prompt-ready string.
    Nothing upstream cares which backend ran.
    """

    @abstractmethod
    async def search(self, query: str, max_results: int) -> list[dict]:
        """
        Returns list of dicts: [{"title": ..., "body": ..., "url": ...}]
        Must never raise — return [] on failure.
        """


# ── backends ──────────────────────────────────────────────────────────────────

class DuckDuckGoBackend(SearchBackend):
    """
    Free, no API key, no account.
    Uses duckduckgo-search package — pip install duckduckgo-search
    Runs in a thread because ddg client is synchronous.
    """

    async def search(self, query: str, max_results: int) -> list[dict]:
        try:
            from ddgs import DDGS
        except ImportError:
            logger.error("duckduckgo-search not installed. Run: pip install duckduckgo-search")
            return []

        def _run():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        try:
            raw = await asyncio.to_thread(_run)
            return [
                {"title": r.get("title",""), "body": r.get("body",""), "url": r.get("href","")}
                for r in raw
            ]
        except Exception as e:
            logger.warning("DuckDuckGo search failed: %s", e)
            return []


class BraveBackend(SearchBackend):
    """
    Brave Search API — free tier 2000 queries/month.
    Needs: BRAVE_API_KEY in .env
    Sign up: brave.com/search/api
    """

    def __init__(self):
        self.api_key = os.getenv("BRAVE_API_KEY", "")
        self.url     = "https://api.search.brave.com/res/v1/web/search"

    async def search(self, query: str, max_results: int) -> list[dict]:
        if not self.api_key:
            logger.error("BRAVE_API_KEY not set in .env")
            return []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    self.url,
                    params={"q": query, "count": max_results},
                    headers={
                        "X-Subscription-Token": self.api_key,
                        "Accept": "application/json",
                    },
                )
                resp.raise_for_status()
                results = resp.json().get("web", {}).get("results", [])
                return [
                    {"title": r.get("title",""), "body": r.get("description",""), "url": r.get("url","")}
                    for r in results
                ]
        except Exception as e:
            logger.warning("Brave search failed: %s", e)
            return []


class TavilyBackend(SearchBackend):
    """
    Tavily — built for AI agents, returns clean structured results.
    Free tier: 1000 searches/month, no credit card.
    Needs: TAVILY_API_KEY in .env
    Sign up: tavily.com
    """

    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY", "")
        self.url     = "https://api.tavily.com/search"

    async def search(self, query: str, max_results: int) -> list[dict]:
        if not self.api_key:
            logger.error("TAVILY_API_KEY not set in .env")
            return []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    self.url,
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "basic",
                    },
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])
                return [
                    {"title": r.get("title",""), "body": r.get("content",""), "url": r.get("url","")}
                    for r in results
                ]
        except Exception as e:
            logger.warning("Tavily search failed: %s", e)
            return []


class SerperBackend(SearchBackend):
    """
    Serper.dev — Google results via API.
    Free tier: 2500 searches, no expiry.
    Needs: SERPER_API_KEY in .env
    Sign up: serper.dev
    """

    def __init__(self):
        self.api_key = os.getenv("SERPER_API_KEY", "")
        self.url     = "https://google.serper.dev/search"

    async def search(self, query: str, max_results: int) -> list[dict]:
        if not self.api_key:
            logger.error("SERPER_API_KEY not set in .env")
            return []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    self.url,
                    headers={
                        "X-API-KEY": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json={"q": query, "num": max_results},
                )
                resp.raise_for_status()
                results = resp.json().get("organic", [])
                return [
                    {"title": r.get("title",""), "body": r.get("snippet",""), "url": r.get("link","")}
                    for r in results
                ]
        except Exception as e:
            logger.warning("Serper search failed: %s", e)
            return []


# ── registry ──────────────────────────────────────────────────────────────────

def _get_backend() -> SearchBackend:
    backends = {
        "duckduckgo": DuckDuckGoBackend,
        "brave":      BraveBackend,
        "tavily":     TavilyBackend,
        "serper":     SerperBackend,
    }
    cls = backends.get(SEARCH_BACKEND)
    if cls is None:
        logger.warning(
            "Unknown SEARCH_BACKEND=%r, falling back to duckduckgo", SEARCH_BACKEND
        )
        cls = DuckDuckGoBackend
    return cls()


# Module-level singleton
_backend: SearchBackend | None = None

def get_backend() -> SearchBackend:
    global _backend
    if _backend is None:
        _backend = _get_backend()
        logger.info("Search backend: %s", SEARCH_BACKEND)
    return _backend


# ── formatter ─────────────────────────────────────────────────────────────────

def _format(results: list[dict], query: str) -> str:
    """
    Turn raw results into a prompt-ready context block.
    Same format regardless of which backend ran.
    """
    if not results:
        return f"No results found for: {query}"

    lines = []
    for r in results:
        title = r.get("title", "")
        body  = r.get("body", "")
        url   = r.get("url", "")
        lines.append(f"- {title}\n  {body}\n  {url}")

    return f"Search results for '{query}':\n\n" + "\n\n".join(lines)


# ── public API (called by registry + executor) ────────────────────────────────

async def web_search(query: str) -> str:
    """
    Search the web using the configured backend.
    Returns a formatted string ready for prompt injection.
    Never raises — returns error string on failure.
    """
    backend = get_backend()
    results = await backend.search(query, max_results=MAX_RESULTS)
    return _format(results, query)
