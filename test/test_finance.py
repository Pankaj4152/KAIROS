"""
Unit tests for the finance tool.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from runtime.tools.finance import finance

@pytest.mark.anyio
async def test_finance_empty_query():
    res = await finance("")
    assert "[FINANCE ERROR: Empty query provided]" in res

@pytest.mark.anyio
async def test_finance_direct_symbol_happy_path():
    chart_data = {
        "chart": {
            "result": [
                {
                    "meta": {
                        "symbol": "AAPL",
                        "shortName": "Apple Inc.",
                        "longName": "Apple Inc.",
                        "currency": "USD",
                        "fullExchangeName": "NasdaqGS",
                        "regularMarketPrice": 175.50,
                        "chartPreviousClose": 173.00,
                        "regularMarketDayLow": 172.50,
                        "regularMarketDayHigh": 176.00,
                        "regularMarketVolume": 52000000,
                        "fiftyTwoWeekLow": 120.00,
                        "fiftyTwoWeekHigh": 198.00
                    }
                }
            ]
        }
    }

    with patch("httpx.AsyncClient") as MockClientClass:
        mock_client = AsyncMock()
        MockClientClass.return_value.__aenter__.return_value = mock_client

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = chart_data
        mock_client.get.return_value = resp

        res = await finance("AAPL", history=False)

        assert "Apple Inc. (AAPL)" in res
        assert "NasdaqGS" in res
        assert "Current Price: 175.50 USD" in res
        assert "Price Change: +2.50 (+1.45%)" in res
        assert "Day's Range: 172.50 - 176.00 USD" in res
        assert "52-Week Range: 120.00 - 198.00 USD" in res
        assert "5-Day Trading History:" not in res

@pytest.mark.anyio
async def test_finance_with_history():
    chart_data = {
        "chart": {
            "result": [
                {
                    "meta": {
                        "symbol": "BTC-USD",
                        "shortName": "Bitcoin USD",
                        "currency": "USD",
                        "fullExchangeName": "CCC",
                        "regularMarketPrice": 62500.0,
                        "chartPreviousClose": 62000.0
                    },
                    "timestamp": [1781481600, 1781568000],
                    "indicators": {
                        "quote": [
                            {
                                "open": [61800.0, 62100.0],
                                "close": [62000.0, 62500.0]
                            }
                        ]
                    }
                }
            ]
        }
    }

    with patch("httpx.AsyncClient") as MockClientClass:
        mock_client = AsyncMock()
        MockClientClass.return_value.__aenter__.return_value = mock_client

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = chart_data
        mock_client.get.return_value = resp

        res = await finance("BTC-USD", history=True)

        assert "Bitcoin USD (BTC-USD)" in res
        assert "Current Price: 62500.00 USD" in res
        assert "5-Day Trading History:" in res
        # Check that we formatted historical dates
        assert "Open 61800.00 | Close 62000.00 USD" in res
        assert "Open 62100.00 | Close 62500.00 USD" in res

@pytest.mark.anyio
async def test_finance_search_resolution_happy_path():
    search_data = {
        "quotes": [
            {
                "symbol": "MSFT",
                "shortname": "Microsoft Corporation",
                "longname": "Microsoft Corporation",
                "exchange": "NMS"
            }
        ]
    }

    chart_data = {
        "chart": {
            "result": [
                {
                    "meta": {
                        "symbol": "MSFT",
                        "longName": "Microsoft Corporation",
                        "currency": "USD",
                        "fullExchangeName": "NasdaqGS",
                        "regularMarketPrice": 420.00,
                        "chartPreviousClose": 418.00
                    }
                }
            ]
        }
    }

    with patch("httpx.AsyncClient") as MockClientClass:
        mock_client = AsyncMock()
        MockClientClass.return_value.__aenter__.return_value = mock_client

        # Mock first direct chart query as failure (404)
        chart_fail = MagicMock()
        chart_fail.status_code = 404

        # Mock search query as success
        search_success = MagicMock()
        search_success.status_code = 200
        search_success.json.return_value = search_data

        # Mock second chart query (with resolved symbol) as success
        chart_success = MagicMock()
        chart_success.status_code = 200
        chart_success.json.return_value = chart_data

        mock_client.get.side_effect = [chart_fail, search_success, chart_success]

        res = await finance("Microsoft", history=False)

        assert "Microsoft Corporation (MSFT)" in res
        assert "Current Price: 420.00 USD" in res
        assert "Price Change: +2.00 (+0.48%)" in res

@pytest.mark.anyio
async def test_finance_resolution_failure():
    with patch("httpx.AsyncClient") as MockClientClass:
        mock_client = AsyncMock()
        MockClientClass.return_value.__aenter__.return_value = mock_client

        # Direct ticker check fails
        chart_fail = MagicMock()
        chart_fail.status_code = 404

        # Search resolution returns no results
        search_fail = MagicMock()
        search_fail.status_code = 200
        search_fail.json.return_value = {"quotes": []}

        mock_client.get.side_effect = [chart_fail, search_fail]

        res = await finance("UnknownCompany")

        assert "[FINANCE ERROR: Could not resolve financial asset for 'UnknownCompany']" in res

@pytest.mark.anyio
async def test_finance_api_exception():
    with patch("httpx.AsyncClient") as MockClientClass:
        mock_client = AsyncMock()
        MockClientClass.return_value.__aenter__.return_value = mock_client

        mock_client.get.side_effect = Exception("Connection timed out")

        res = await finance("AAPL")

        assert "[FINANCE ERROR: Yahoo Finance API failed" in res
        assert "Connection timed out" in res
