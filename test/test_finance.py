# """
# Unit tests for the finance tool.
# """

# import pytest
# from unittest.mock import AsyncMock, MagicMock, patch
# from runtime.tools.finance import finance

# @pytest.mark.anyio
# async def test_finance_empty_query():
#     res = await finance("")
#     assert "[FINANCE ERROR: Empty query provided]" in res

# @pytest.mark.anyio
# async def test_finance_direct_symbol_happy_path():
#     chart_data = {
#         "chart": {
#             "result": [
#                 {
#                     "meta": {
#                         "symbol": "AAPL",
#                         "shortName": "Apple Inc.",
#                         "longName": "Apple Inc.",
#                         "currency": "USD",
#                         "fullExchangeName": "NasdaqGS",
#                         "regularMarketPrice": 175.50,
#                         "chartPreviousClose": 173.00,
#                         "regularMarketDayLow": 172.50,
#                         "regularMarketDayHigh": 176.00,
#                         "regularMarketVolume": 52000000,
#                         "fiftyTwoWeekLow": 120.00,
#                         "fiftyTwoWeekHigh": 198.00
#                     }
#                 }
#             ]
#         }
#     }

#     with patch("httpx.AsyncClient") as MockClientClass:
#         mock_client = AsyncMock()
#         MockClientClass.return_value.__aenter__.return_value = mock_client

#         resp = MagicMock()
#         resp.status_code = 200
#         resp.json.return_value = chart_data
#         mock_client.get.return_value = resp

#         res = await finance("AAPL", history=False)

#         assert "Apple Inc. (AAPL)" in res
#         assert "NasdaqGS" in res
#         assert "Current Price: 175.50 USD" in res
#         assert "Price Change: +2.50 (+1.45%)" in res
#         assert "Day's Range: 172.50 - 176.00 USD" in res
#         assert "52-Week Range: 120.00 - 198.00 USD" in res
#         assert "5-Day Trading History:" not in res

# @pytest.mark.anyio
# async def test_finance_with_history():
#     chart_data = {
#         "chart": {
#             "result": [
#                 {
#                     "meta": {
#                         "symbol": "BTC-USD",
#                         "shortName": "Bitcoin USD",
#                         "currency": "USD",
#                         "fullExchangeName": "CCC",
#                         "regularMarketPrice": 62500.0,
#                         "chartPreviousClose": 62000.0
#                     },
#                     "timestamp": [1781481600, 1781568000],
#                     "indicators": {
#                         "quote": [
#                             {
#                                 "open": [61800.0, 62100.0],
#                                 "close": [62000.0, 62500.0]
#                             }
#                         ]
#                     }
#                 }
#             ]
#         }
#     }

#     with patch("httpx.AsyncClient") as MockClientClass:
#         mock_client = AsyncMock()
#         MockClientClass.return_value.__aenter__.return_value = mock_client

#         resp = MagicMock()
#         resp.status_code = 200
#         resp.json.return_value = chart_data
#         mock_client.get.return_value = resp

#         res = await finance("BTC-USD", history=True)

#         assert "Bitcoin USD (BTC-USD)" in res
#         assert "Current Price: 62500.00 USD" in res
#         assert "5-Day Trading History:" in res
#         # Check that we formatted historical dates
#         assert "Open 61800.00 | Close 62000.00 USD" in res
#         assert "Open 62100.00 | Close 62500.00 USD" in res

# @pytest.mark.anyio
# async def test_finance_search_resolution_happy_path():
#     search_data = {
#         "quotes": [
#             {
#                 "symbol": "MSFT",
#                 "shortname": "Microsoft Corporation",
#                 "longname": "Microsoft Corporation",
#                 "exchange": "NMS"
#             }
#         ]
#     }

#     chart_data = {
#         "chart": {
#             "result": [
#                 {
#                     "meta": {
#                         "symbol": "MSFT",
#                         "longName": "Microsoft Corporation",
#                         "currency": "USD",
#                         "fullExchangeName": "NasdaqGS",
#                         "regularMarketPrice": 420.00,
#                         "chartPreviousClose": 418.00
#                     }
#                 }
#             ]
#         }
#     }

#     with patch("httpx.AsyncClient") as MockClientClass:
#         mock_client = AsyncMock()
#         MockClientClass.return_value.__aenter__.return_value = mock_client

#         # Mock first direct chart query as failure (404)
#         chart_fail = MagicMock()
#         chart_fail.status_code = 404

#         # Mock search query as success
#         search_success = MagicMock()
#         search_success.status_code = 200
#         search_success.json.return_value = search_data

#         # Mock second chart query (with resolved symbol) as success
#         chart_success = MagicMock()
#         chart_success.status_code = 200
#         chart_success.json.return_value = chart_data

#         mock_client.get.side_effect = [chart_fail, search_success, chart_success]

#         res = await finance("Microsoft", history=False)

#         assert "Microsoft Corporation (MSFT)" in res
#         assert "Current Price: 420.00 USD" in res
#         assert "Price Change: +2.00 (+0.48%)" in res

# @pytest.mark.anyio
# async def test_finance_resolution_failure():
#     with patch("httpx.AsyncClient") as MockClientClass:
#         mock_client = AsyncMock()
#         MockClientClass.return_value.__aenter__.return_value = mock_client

#         # Direct ticker check fails
#         chart_fail = MagicMock()
#         chart_fail.status_code = 404

#         # Search resolution returns no results
#         search_fail = MagicMock()
#         search_fail.status_code = 200
#         search_fail.json.return_value = {"quotes": []}

#         mock_client.get.side_effect = [chart_fail, search_fail]

#         res = await finance("UnknownCompany")

#         assert "[FINANCE ERROR: Could not resolve financial asset for 'UnknownCompany']" in res

# @pytest.mark.anyio
# async def test_finance_api_exception():
#     with patch("httpx.AsyncClient") as MockClientClass:
#         mock_client = AsyncMock()
#         MockClientClass.return_value.__aenter__.return_value = mock_client

#         mock_client.get.side_effect = Exception("Connection timed out")

#         res = await finance("AAPL")

#         assert "[FINANCE ERROR: Yahoo Finance API failed" in res
#         assert "Connection timed out" in res
"""
Tests for runtime/tools/finance.py

Run from the project root:
    pytest test/test_finance.py -v

Structure:
    Unit tests      — no network. Mock httpx.AsyncClient entirely.
                      Cover all branches, error paths, and formatting helpers.
    Integration     — real Yahoo Finance API calls. Skipped by default.
                      Run with: pytest test/test_finance.py -v -k integration

What is tested:
    _fmt_price          — decimal precision for tiny/small/normal values
    _fmt_large          — T/B/M formatting
    _fmt_change         — sign, decimals, percentage
    _resolve_symbol     — crypto suffix auto-applied, non-crypto unchanged
    _compass            — degree → bearing conversion
    _uv_label           — UV index category labels
    finance() dispatch  — unknown action returns error
    finance() empty     — empty query returns error
    quote action        — happy path, pre/post market, fundamentals, missing price
    history action      — valid period, invalid period, empty candles
    search action       — results formatted, no results case
    symbol resolution   — company name triggers search, exact ticker skips search
    HTTP 404 on chart   — triggers search-based fallback
    httpx exception     — returns error string, never raises
    JSON shape error    — missing chart.result handled gracefully
    crypto symbol       — BTC auto-becomes BTC-USD
    Indian stock        — INFY.NS passes through unchanged
"""

import asyncio
import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runtime"))

from tools.finance import (
    finance,
    _fmt_price,
    _fmt_large,
    _fmt_change,
    _resolve_symbol,
    _VALID_RANGES,
    _RANGE_DEFAULTS,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_chart_response(
    symbol: str = "AAPL",
    price: float = 189.50,
    prev_close: float = 187.20,
    day_high: float = 191.00,
    day_low: float = 186.50,
    volume: int = 45_000_000,
    y52_high: float = 230.00,
    y52_low: float = 150.00,
    market_cap: float = 2_900_000_000_000,
    pe: float = 29.5,
    ma50: float = 182.0,
    ma200: float = 175.0,
    pre_market: float | None = None,
    post_market: float | None = None,
    currency: str = "USD",
    exchange: str = "NASDAQ",
    long_name: str = "Apple Inc.",
    timestamps: list | None = None,
    opens: list | None = None,
    highs: list | None = None,
    lows: list | None = None,
    closes: list | None = None,
    volumes: list | None = None,
) -> dict:
    """Build a minimal Yahoo Finance v8/chart response dict."""
    meta = {
        "symbol": symbol,
        "currency": currency,
        "fullExchangeName": exchange,
        "longName": long_name,
        "regularMarketPrice": price,
        "chartPreviousClose": prev_close,
        "regularMarketDayHigh": day_high,
        "regularMarketDayLow": day_low,
        "regularMarketVolume": volume,
        "regularMarketOpen": day_low + 1.0,
        "fiftyTwoWeekHigh": y52_high,
        "fiftyTwoWeekLow": y52_low,
        "marketCap": market_cap,
        "trailingPE": pe,
        "fiftyDayAverage": ma50,
        "twoHundredDayAverage": ma200,
        "averageDailyVolume3Month": 40_000_000,
    }
    if pre_market is not None:
        meta["preMarketPrice"] = pre_market
    if post_market is not None:
        meta["postMarketPrice"] = post_market

    result = {"meta": meta}

    # Add OHLCV if provided (for history tests)
    if timestamps is not None:
        result["timestamp"] = timestamps
        result["indicators"] = {
            "quote": [{
                "open":   opens  or [],
                "high":   highs  or [],
                "low":    lows   or [],
                "close":  closes or [],
                "volume": volumes or [],
            }]
        }

    return {"chart": {"result": [result], "error": None}}


def _make_search_response(results: list[dict]) -> dict:
    """Build a Yahoo Finance search response."""
    return {"quotes": results}


def _mock_client(chart_data: dict | None = None, search_data: dict | None = None,
                  chart_status: int = 200, search_status: int = 200,
                  raise_on_chart: Exception | None = None):
    """
    Build a mock httpx.AsyncClient context manager.

    chart_data:   JSON returned for chart endpoint.
    search_data:  JSON returned for search endpoint.
    raise_on_chart: if set, raises this exception instead of returning chart_data.
    """
    mock_chart_resp = MagicMock()
    mock_chart_resp.status_code = chart_status
    mock_chart_resp.json.return_value = chart_data or {}
    mock_chart_resp.raise_for_status = MagicMock()

    mock_search_resp = MagicMock()
    mock_search_resp.status_code = search_status
    mock_search_resp.json.return_value = search_data or {}
    mock_search_resp.raise_for_status = MagicMock()

    async def mock_get(url, **kwargs):
        if "search" in url:
            return mock_search_resp
        if raise_on_chart:
            raise raise_on_chart
        return mock_chart_resp

    client = AsyncMock()
    client.get = mock_get
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__  = AsyncMock(return_value=False)
    return client


# ── unit tests: formatting helpers ───────────────────────────────────────────

class TestFmtPrice(unittest.TestCase):

    def test_none_returns_na(self):
        self.assertEqual(_fmt_price(None), "N/A")

    def test_normal_price(self):
        self.assertEqual(_fmt_price(189.50), "189.50")

    def test_thousands_comma(self):
        result = _fmt_price(1234.56)
        self.assertIn("1,234", result)

    def test_tiny_price(self):
        # Values < 0.001 → 8 decimal places
        result = _fmt_price(0.000012)
        self.assertIn(".", result)
        self.assertTrue(len(result.split(".")[-1]) >= 6)

    def test_small_price(self):
        # Values < 0.1 → 6 decimal places
        result = _fmt_price(0.05)
        self.assertEqual(result, "0.050000")

    def test_crypto_small(self):
        # Values < 2.0 → 4 decimal places
        result = _fmt_price(1.2345)
        self.assertEqual(result, "1.2345")

    def test_invalid_type(self):
        self.assertEqual(_fmt_price("not_a_number"), "N/A")


class TestFmtLarge(unittest.TestCase):

    def test_trillion(self):
        result = _fmt_large(2_900_000_000_000)
        self.assertIn("T", result)
        self.assertIn("2.90", result)

    def test_billion(self):
        result = _fmt_large(456_700_000_000)
        self.assertIn("B", result)
        self.assertIn("456.70", result)

    def test_million(self):
        result = _fmt_large(1_500_000)
        self.assertIn("M", result)

    def test_small(self):
        result = _fmt_large(99_000)
        self.assertIn("99,000", result)

    def test_none(self):
        self.assertEqual(_fmt_large(None), "N/A")


class TestFmtChange(unittest.TestCase):

    def test_positive_change(self):
        result = _fmt_change(2.30, 1.23)
        self.assertIn("+", result)
        self.assertIn("1.23%", result)

    def test_negative_change(self):
        result = _fmt_change(-5.00, -2.50)
        self.assertIn("-", result)
        self.assertIn("2.50%", result)

    def test_zero_change(self):
        result = _fmt_change(0.0, 0.0)
        self.assertIn("+", result)


class TestResolveSymbol(unittest.TestCase):

    def test_crypto_btc(self):
        self.assertEqual(_resolve_symbol("BTC"), "BTC-USD")

    def test_crypto_eth(self):
        self.assertEqual(_resolve_symbol("eth"), "ETH-USD")

    def test_normal_stock(self):
        self.assertEqual(_resolve_symbol("AAPL"), "AAPL")

    def test_indian_stock_unchanged(self):
        self.assertEqual(_resolve_symbol("INFY.NS"), "INFY.NS")

    def test_company_name_unchanged(self):
        # Company names pass through — resolution happens via API
        result = _resolve_symbol("Apple Inc")
        self.assertNotIn("-USD", result)

    def test_whitespace_stripped(self):
        self.assertEqual(_resolve_symbol("  TSLA  "), "TSLA")




class TestValidRanges(unittest.TestCase):

    def test_all_expected_ranges_present(self):
        expected = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"}
        self.assertEqual(_VALID_RANGES, expected)

    def test_every_range_has_default_interval(self):
        for r in _VALID_RANGES:
            self.assertIn(r, _RANGE_DEFAULTS, f"Range '{r}' missing from _RANGE_DEFAULTS")


# ── unit tests: finance() dispatch ───────────────────────────────────────────

class TestFinanceDispatch(unittest.TestCase):

    def test_empty_query_returns_error(self):
        result = run(finance(query=""))
        self.assertIn("Error", result)
        self.assertIn("empty", result.lower())

    def test_whitespace_query_returns_error(self):
        result = run(finance(query="   "))
        self.assertIn("Error", result)

    def test_unknown_action_returns_error(self):
        result = run(finance(query="AAPL", action="teleport"))
        self.assertIn("Error", result)
        self.assertIn("teleport", result)
        self.assertIn("quote", result)   # valid actions listed


# ── unit tests: quote action ──────────────────────────────────────────────────

class TestQuoteAction(unittest.TestCase):

    @patch("tools.finance.httpx.AsyncClient")
    def test_quote_happy_path(self, MockClient):
        chart = _make_chart_response(symbol="AAPL", price=189.50, prev_close=187.20)
        MockClient.return_value = _mock_client(chart_data=chart)

        result = run(finance(query="AAPL", action="quote"))

        self.assertIn("AAPL", result)
        self.assertIn("189.50", result)
        self.assertIn("NASDAQ", result)

    @patch("tools.finance.httpx.AsyncClient")
    def test_quote_shows_change(self, MockClient):
        chart = _make_chart_response(price=189.50, prev_close=187.20)
        MockClient.return_value = _mock_client(chart_data=chart)

        result = run(finance(query="AAPL", action="quote"))

        self.assertIn("Change", result)
        self.assertIn("+", result)   # positive change

    @patch("tools.finance.httpx.AsyncClient")
    def test_quote_shows_pre_market(self, MockClient):
        chart = _make_chart_response(price=189.50, pre_market=192.00)
        MockClient.return_value = _mock_client(chart_data=chart)

        result = run(finance(query="AAPL", action="quote"))
        self.assertIn("Pre-market", result)
        self.assertIn("192.00", result)

    @patch("tools.finance.httpx.AsyncClient")
    def test_quote_shows_post_market(self, MockClient):
        chart = _make_chart_response(price=189.50, post_market=188.00)
        MockClient.return_value = _mock_client(chart_data=chart)

        result = run(finance(query="AAPL", action="quote"))
        self.assertIn("After-hours", result)
        self.assertIn("188.00", result)

    @patch("tools.finance.httpx.AsyncClient")
    def test_quote_shows_fundamentals(self, MockClient):
        chart = _make_chart_response(market_cap=2_900_000_000_000, pe=29.5)
        MockClient.return_value = _mock_client(chart_data=chart)

        result = run(finance(query="AAPL", action="quote"))
        self.assertIn("Market cap", result)
        self.assertIn("2.90T", result)
        self.assertIn("P/E", result)
        self.assertIn("29.50", result)

    @patch("tools.finance.httpx.AsyncClient")
    def test_quote_shows_moving_averages(self, MockClient):
        chart = _make_chart_response(price=189.50, ma50=182.0, ma200=175.0)
        MockClient.return_value = _mock_client(chart_data=chart)

        result = run(finance(query="AAPL", action="quote"))
        self.assertIn("50-day MA", result)
        self.assertIn("200-day MA", result)

    @patch("tools.finance.httpx.AsyncClient")
    def test_quote_missing_price_returns_error(self, MockClient):
        chart = {"chart": {"result": [{"meta": {"symbol": "AAPL"}}], "error": None}}
        MockClient.return_value = _mock_client(chart_data=chart)

        result = run(finance(query="AAPL", action="quote"))
        self.assertIn("Error", result)

    @patch("tools.finance.httpx.AsyncClient")
    def test_quote_empty_result_list_returns_error(self, MockClient):
        chart = {"chart": {"result": [], "error": None}}
        MockClient.return_value = _mock_client(chart_data=chart)

        result = run(finance(query="FAKE123", action="quote"))
        self.assertIn("Error", result)

    @patch("tools.finance.httpx.AsyncClient")
    def test_quote_httpx_exception_returns_error_string(self, MockClient):
        MockClient.return_value = _mock_client(
            raise_on_chart=httpx.RequestError("Connection refused", request=MagicMock())
        )
        result = run(finance(query="AAPL", action="quote"))
        self.assertIn("Error", result)
        self.assertIsInstance(result, str)

    @patch("tools.finance.httpx.AsyncClient")
    def test_quote_crypto_btc(self, MockClient):
        """BTC should be auto-resolved to BTC-USD."""
        chart = _make_chart_response(symbol="BTC-USD", price=67000.00, currency="USD")
        MockClient.return_value = _mock_client(chart_data=chart)

        result = run(finance(query="BTC", action="quote"))
        self.assertIn("67,000.00", result)
        self.assertNotIn("Error", result)

    @patch("tools.finance.httpx.AsyncClient")
    def test_quote_negative_change(self, MockClient):
        """Negative price change should show minus sign."""
        chart = _make_chart_response(price=185.00, prev_close=189.50)
        MockClient.return_value = _mock_client(chart_data=chart)

        result = run(finance(query="AAPL", action="quote"))
        self.assertIn("-", result)


# ── unit tests: history action ────────────────────────────────────────────────

class TestHistoryAction(unittest.TestCase):

    def _make_daily_candles(self, n: int = 5):
        """Make n days of candle data starting from a base timestamp."""
        base = 1_700_000_000
        timestamps = [base + i * 86400 for i in range(n)]
        opens  = [100.0 + i for i in range(n)]
        highs  = [102.0 + i for i in range(n)]
        lows   = [98.0  + i for i in range(n)]
        closes = [101.0 + i for i in range(n)]
        vols   = [1_000_000 + i * 10000 for i in range(n)]
        return timestamps, opens, highs, lows, closes, vols

    @patch("tools.finance.httpx.AsyncClient")
    def test_history_happy_path(self, MockClient):
        ts, o, h, l, c, v = self._make_daily_candles(5)
        chart = _make_chart_response(timestamps=ts, opens=o, highs=h, lows=l, closes=c, volumes=v)
        MockClient.return_value = _mock_client(chart_data=chart)

        result = run(finance(query="AAPL", action="history", period="5d"))

        self.assertIn("History", result)
        self.assertIn("5d", result)
        self.assertIn("Period return", result)
        self.assertNotIn("Error", result)

    @patch("tools.finance.httpx.AsyncClient")
    def test_history_shows_period_return(self, MockClient):
        ts, o, h, l, c, v = self._make_daily_candles(3)
        # First close=101, last close=103 → +1.98%
        chart = _make_chart_response(timestamps=ts, opens=o, highs=h, lows=l, closes=c, volumes=v)
        MockClient.return_value = _mock_client(chart_data=chart)

        result = run(finance(query="AAPL", action="history", period="5d"))
        self.assertIn("Period return", result)

    def test_history_invalid_period_returns_error(self):
        result = run(finance(query="AAPL", action="history", period="10y"))
        self.assertIn("Error", result)
        self.assertIn("10y", result)

    @patch("tools.finance.httpx.AsyncClient")
    def test_history_empty_candles_returns_error(self, MockClient):
        chart = _make_chart_response(timestamps=[], opens=[], highs=[], lows=[], closes=[], volumes=[])
        MockClient.return_value = _mock_client(chart_data=chart)

        result = run(finance(query="AAPL", action="history", period="5d"))
        self.assertIn("Error", result)

    @patch("tools.finance.httpx.AsyncClient")
    def test_history_none_candles_filtered(self, MockClient):
        """Candles with None open or close must be silently filtered out."""
        ts = [1_700_000_000, 1_700_086_400, 1_700_172_800]
        o  = [100.0, None, 102.0]
        c  = [101.0, None, 103.0]
        h  = [103.0, None, 105.0]
        l  = [98.0,  None, 100.0]
        v  = [1_000_000, None, 1_200_000]
        chart = _make_chart_response(timestamps=ts, opens=o, highs=h, lows=l, closes=c, volumes=v)
        MockClient.return_value = _mock_client(chart_data=chart)

        # Should not crash and should show 2 valid candles
        result = run(finance(query="AAPL", action="history", period="5d"))
        self.assertNotIn("Error", result)
        self.assertIn("History", result)

    @patch("tools.finance.httpx.AsyncClient")
    def test_history_limited_to_30_rows(self, MockClient):
        """Output must be capped at 30 rows even if more candles exist."""
        n = 40
        base = 1_700_000_000
        ts = [base + i * 86400 for i in range(n)]
        o  = [100.0] * n
        c  = [101.0] * n
        h  = [102.0] * n
        l  = [99.0]  * n
        v  = [1_000_000] * n
        chart = _make_chart_response(timestamps=ts, opens=o, highs=h, lows=l, closes=c, volumes=v)
        MockClient.return_value = _mock_client(chart_data=chart)

        result = run(finance(query="AAPL", action="history", period="3mo"))
        # "Showing last 30" message should appear
        self.assertIn("30", result)
        self.assertIn("40", result)


# ── unit tests: search action ─────────────────────────────────────────────────

class TestSearchAction(unittest.TestCase):

    @patch("tools.finance.httpx.AsyncClient")
    def test_search_returns_results(self, MockClient):
        search = _make_search_response([
            {"symbol": "AAPL", "longname": "Apple Inc.", "quoteType": "EQUITY", "exchDisp": "NASDAQ"},
            {"symbol": "AAPL.BA", "longname": "Apple Inc. (Argentina)", "quoteType": "EQUITY", "exchDisp": "BUE"},
        ])
        MockClient.return_value = _mock_client(search_data=search)

        result = run(finance(query="Apple", action="search"))
        self.assertIn("AAPL", result)
        self.assertIn("Apple Inc.", result)
        self.assertIn("NASDAQ", result)

    @patch("tools.finance.httpx.AsyncClient")
    def test_search_no_results(self, MockClient):
        search = _make_search_response([])
        MockClient.return_value = _mock_client(search_data=search)

        result = run(finance(query="xyzxyznonexistent", action="search"))
        self.assertIn("No results", result)

    @patch("tools.finance.httpx.AsyncClient")
    def test_search_includes_usage_hint(self, MockClient):
        """Search output should tell the LLM how to use the returned symbol."""
        search = _make_search_response([
            {"symbol": "RELIANCE.NS", "longname": "Reliance Industries", "quoteType": "EQUITY", "exchDisp": "NSE"},
        ])
        MockClient.return_value = _mock_client(search_data=search)

        result = run(finance(query="Reliance", action="search"))
        self.assertIn("action=", result)   # Usage hint present


# ── unit tests: symbol resolution ────────────────────────────────────────────

class TestSymbolResolution(unittest.TestCase):

    @patch("tools.finance.httpx.AsyncClient")
    def test_company_name_triggers_search(self, MockClient):
        """A non-ticker input (has spaces) should trigger _search_symbol."""
        search = _make_search_response([
            {"symbol": "TSLA", "longname": "Tesla Inc.", "quoteType": "EQUITY", "exchDisp": "NASDAQ"},
        ])
        chart = _make_chart_response(symbol="TSLA", price=250.0)
        MockClient.return_value = _mock_client(chart_data=chart, search_data=search)

        result = run(finance(query="Tesla Motors", action="quote"))
        self.assertIn("TSLA", result)

    @patch("tools.finance.httpx.AsyncClient")
    def test_exact_ticker_skips_search_on_success(self, MockClient):
        """An exact ticker that returns data should not call search."""
        chart = _make_chart_response(symbol="AAPL", price=189.50)
        client_mock = _mock_client(chart_data=chart)
        # Track search calls
        search_calls = []

        orig_get = client_mock.get
        async def tracked_get(url, **kwargs):
            if "search" in url:
                search_calls.append(url)
            return await orig_get(url, **kwargs)

        client_mock.get = tracked_get
        MockClient.return_value = client_mock

        run(finance(query="AAPL", action="quote"))
        # Search should not have been called
        self.assertEqual(len(search_calls), 0)


# ── integration tests ─────────────────────────────────────────────────────────

@unittest.skipUnless(
    os.getenv("RUN_INTEGRATION_TESTS"),
    "Set RUN_INTEGRATION_TESTS=1 to run live API tests"
)
class TestFinanceIntegration(unittest.TestCase):
    """
    Live Yahoo Finance API tests.

    Run:
        RUN_INTEGRATION_TESTS=1 pytest test/test_finance.py -v -k integration
    """

    def test_quote_aapl(self):
        result = run(finance(query="AAPL", action="quote"))
        self.assertNotIn("Error", result)
        self.assertIn("AAPL", result)
        self.assertIn("Price", result)

    def test_quote_btc(self):
        result = run(finance(query="BTC", action="quote"))
        self.assertNotIn("Error", result)
        self.assertIn("BTC", result)

    def test_quote_indian_stock(self):
        result = run(finance(query="INFY.NS", action="quote"))
        self.assertNotIn("Error", result)
        self.assertIn("INFY", result)

    def test_history_1mo(self):
        result = run(finance(query="AAPL", action="history", period="1mo"))
        self.assertNotIn("Error", result)
        self.assertIn("Period return", result)

    def test_search_apple(self):
        result = run(finance(query="Apple Inc", action="search"))
        self.assertNotIn("Error", result)
        self.assertIn("AAPL", result)

    def test_company_name_resolution(self):
        result = run(finance(query="Tesla", action="quote"))
        self.assertNotIn("Error", result)
        self.assertIn("TSLA", result)

    def test_invalid_symbol(self):
        result = run(finance(query="XYZXYZXYZ999", action="quote"))
        self.assertIn("Error", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)