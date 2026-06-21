# """
# Finance tool — retrieves stock and cryptocurrency pricing, key statistics, and historical trends.
# Uses Yahoo Finance keyless API endpoints. Optimized for KAIROS.
# """

# import logging
# import datetime
# import httpx

# logger = logging.getLogger(__name__)

# HEADERS = {
#     "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
#     "Accept": "text/html,application/json,application/xhtml+xml,application/xml;q=0.9",
#     "Accept-Language": "en-US,en;q=0.9"
# }

# # Common crypto abbreviations to automatically suffix with -USD
# CRYPTO_SYMBOLS = {
#     "BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "DOT", "UNI", "LINK", "LTC", "BCH",
#     "XLM", "ETC", "FIL", "VET", "TRX", "EOS", "XTZ", "AAVE", "MKR", "COMP", "YFI"
# }

# async def _resolve_ticker(query: str) -> tuple[str, str]:
#     search_url = "https://query2.finance.yahoo.com/v1/finance/search"
#     try:
#         async with httpx.AsyncClient(headers=HEADERS, timeout=10.0, follow_redirects=True) as client:
#             resp = await client.get(search_url, params={"q": query, "quotesCount": 1})
#             if resp.status_code == 200:
#                 data = resp.json()
#                 quotes = data.get("quotes", [])
#                 if quotes:
#                     best_match = quotes[0]
#                     symbol = best_match.get("symbol", query)
#                     name = best_match.get("longname") or best_match.get("shortname") or symbol
#                     return symbol, name
#     except Exception as e:
#         logger.warning("Failed to resolve ticker for '%s': %s", query, e)
#     return query, ""

# async def finance(query: str, history: bool = False) -> str:
#     query = query.strip()
#     if not query:
#         return "[FINANCE ERROR: Empty query provided]"

#     symbol = query
#     if symbol.upper() in CRYPTO_SYMBOLS:
#         symbol = f"{symbol.upper()}-USD"

#     asset_name = ""
#     chart_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
#     params = {"interval": "1d", "range": "5d" if history else "1d"}
    
#     try:
#         async with httpx.AsyncClient(headers=HEADERS, timeout=10.0, follow_redirects=True) as client:
#             resp = await client.get(chart_url, params=params)
            
#             if resp.status_code != 200:
#                 resolved_symbol, resolved_name = await _resolve_ticker(query)
#                 if resolved_symbol.upper() != symbol.upper():
#                     symbol = resolved_symbol
#                     asset_name = resolved_name
#                     chart_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
#                     resp = await client.get(chart_url, params=params)
                    
#             if resp.status_code != 200:
#                 return f"[FINANCE ERROR: Could not resolve financial asset for '{query}']"
                
#             chart_data = resp.json()
#     except Exception as e:
#         logger.exception("Yahoo Finance API request failed for: %s", query)
#         return f"[FINANCE ERROR: Yahoo Finance API failed — {type(e).__name__}: {str(e)}]"

#     result_list = chart_data.get("chart", {}).get("result", [])
#     if not result_list:
#         return f"[FINANCE ERROR: No data found for symbol '{symbol}']"

#     res = result_list[0]
#     meta = res.get("meta", {})
    
#     current_price = meta.get("regularMarketPrice")
#     prev_close = meta.get("chartPreviousClose")
#     currency = meta.get("currency", "USD")
#     exchange = meta.get("fullExchangeName") or meta.get("exchangeName", "Unknown")
#     symbol = meta.get("symbol", symbol)
#     long_name = asset_name or meta.get("longName") or meta.get("shortName") or symbol

#     if current_price is None:
#         return f"[FINANCE ERROR: Could not retrieve current price for symbol '{symbol}']"

#     def fmt_price(val) -> str:
#         if val is None: return "N/A"
#         if val < 0.1: return f"{val:.6f}"
#         if val < 2.0: return f"{val:.4f}"
#         return f"{val:.2f}"

#     lines = [
#         f"Financial Data for: {long_name} ({symbol})",
#         f"Exchange: {exchange} | Currency: {currency}",
#         f"Current Price: {fmt_price(current_price)} {currency}"
#     ]

#     if prev_close is not None:
#         change = current_price - prev_close
#         pct_change = (change / prev_close) * 100 if prev_close != 0 else 0.0
#         sign = "+" if change >= 0 else ""
#         lines.append(f"Price Change: {sign}{fmt_price(change)} ({sign}{pct_change:.2f}%) vs Previous Close ({fmt_price(prev_close)} {currency})")

#     day_high = meta.get("regularMarketDayHigh")
#     day_low = meta.get("regularMarketDayLow")
#     volume = meta.get("regularMarketVolume")
#     y52_high = meta.get("fiftyTwoWeekHigh")
#     y52_low = meta.get("fiftyTwoWeekLow")

#     lines.append("Key Statistics:")
#     if day_low is not None and day_high is not None:
#         lines.append(f"  - Day's Range: {fmt_price(day_low)} - {fmt_price(day_high)} {currency}")
#     if volume is not None:
#         lines.append(f"  - Day's Volume: {volume:,}")
#     if y52_low is not None and y52_high is not None:
#         lines.append(f"  - 52-Week Range: {fmt_price(y52_low)} - {fmt_price(y52_high)} {currency}")

#     if history:
#         timestamps = res.get("timestamp", [])
#         indicators = res.get("indicators", {}).get("quote", [{}])
        
#         if indicators and timestamps:
#             quote = indicators[0]
#             opens = quote.get("open", [])
#             closes = quote.get("close", [])
            
#             history_data = list(zip(timestamps, opens, closes))
#             # Filter systematically before array index slice allocations
#             history_data = [(t, o, c) for t, o, c in history_data if o is not None and c is not None]
            
#             if history_data:
#                 lines.append("\n5-Day Trading History:")
#                 for t, o, c in history_data[-5:]:
#                     try:
#                         date_str = datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).strftime('%Y-%m-%d')
#                     except Exception:
#                         date_str = "N/A"
#                     lines.append(f"  - {date_str}: Open {fmt_price(o)} | Close {fmt_price(c)} {currency}")

#     return "\n".join(lines)
"""
Finance tool — stock and cryptocurrency pricing, fundamentals, and historical data.

Uses Yahoo Finance public API endpoints (no API key required).

Actions:
    quote       — current price, change, day range, volume, 52-week range,
                  market cap, P/E, moving averages, pre/post-market price (default)
    history     — OHLCV candles for a configurable period and interval
    search      — resolve a company name or partial ticker to a symbol

Design rules (matching the rest of Kairos tooling):
    - One shared httpx.AsyncClient per call — no per-request client creation.
    - Returns plain strings always. Never raises to the LLM.
    - Errors are plain "Error: ..." strings, not bracketed [ERROR] prefixes.
    - Crypto symbols auto-detected and suffixed with -USD.
    - Symbol resolution (company name → ticker) runs on every non-exact match,
      not only on HTTP failures.
    - Market cap and large numbers formatted for readability (1.23T, 456.7B, etc.)

Env vars required: none (Yahoo Finance is keyless).
"""

import asyncio
import datetime
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://finance.yahoo.com",
    "Referer": "https://finance.yahoo.com/",
}

# Known crypto base symbols → auto-appended with -USD
# LLM can say "BTC" and we pass "BTC-USD" to Yahoo Finance
_CRYPTO_SYMBOLS = frozenset({
    "BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "DOT", "UNI", "LINK",
    "LTC", "BCH", "XLM", "ETC", "FIL", "VET", "TRX", "EOS", "XTZ",
    "AAVE", "MKR", "COMP", "YFI", "AVAX", "MATIC", "ATOM", "NEAR",
    "FTM", "ALGO", "ICP", "SHIB", "TON", "SUI", "APT", "SEI",
})

# Yahoo Finance history: valid range values and their sensible default intervals
# Interval is chosen automatically based on range to keep response concise.
_RANGE_DEFAULTS: dict[str, str] = {
    "1d":  "5m",
    "5d":  "1d",
    "1mo": "1d",
    "3mo": "1d",
    "6mo": "1wk",
    "1y":  "1wk",
    "2y":  "1mo",
    "5y":  "1mo",
    "max": "1mo",
}

_VALID_RANGES = set(_RANGE_DEFAULTS.keys())

_YF_CHART_URL  = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_YF_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_price(val: Any) -> str:
    """Format a price value with appropriate decimal places."""
    if val is None:
        return "N/A"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if v < 0.001:
        return f"{v:.8f}"
    if v < 0.1:
        return f"{v:.6f}"
    if v < 2.0:
        return f"{v:.4f}"
    return f"{v:,.2f}"


def _fmt_large(val: Any) -> str:
    """Format large numbers as T/B/M with one decimal place."""
    if val is None:
        return "N/A"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if v >= 1e12:
        return f"{v / 1e12:.2f}T"
    if v >= 1e9:
        return f"{v / 1e9:.2f}B"
    if v >= 1e6:
        return f"{v / 1e6:.2f}M"
    return f"{v:,.0f}"


def _fmt_change(change: float, pct: float) -> str:
    sign = "+" if change >= 0 else ""
    return f"{sign}{_fmt_price(change)} ({sign}{pct:.2f}%)"


def _resolve_symbol(raw: str) -> str:
    """
    Apply crypto suffix if needed.
    Does NOT query the API — that's done in _search_symbol() if necessary.
    """
    upper = raw.strip().upper()
    if upper in _CRYPTO_SYMBOLS:
        return f"{upper}-USD"
    return raw.strip()


def _ts_to_date(ts: int) -> str:
    try:
        return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return "N/A"


def _ts_to_datetime(ts: int) -> str:
    try:
        return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return "N/A"


# ── API calls ─────────────────────────────────────────────────────────────────

async def _search_symbol(query: str, client: httpx.AsyncClient) -> tuple[str, str]:
    """
    Resolve a company name or ambiguous query to a (symbol, display_name) tuple.
    Returns (query, "") if resolution fails — caller keeps original input.
    """
    try:
        resp = await client.get(
            _YF_SEARCH_URL,
            params={"q": query, "quotesCount": 3, "newsCount": 0},
        )
        if resp.status_code == 200:
            quotes = resp.json().get("quotes", [])
            if quotes:
                best = quotes[0]
                sym  = best.get("symbol", query)
                name = best.get("longname") or best.get("shortname") or sym
                return sym, name
    except Exception as e:
        logger.debug("Symbol search failed for %r: %s", query, e)
    return query, ""


async def _fetch_chart(
    symbol: str,
    range_: str,
    interval: str,
    client: httpx.AsyncClient,
) -> dict | None:
    """Fetch Yahoo Finance chart data. Returns parsed dict or None on failure."""
    try:
        resp = await client.get(
            _YF_CHART_URL.format(symbol=symbol),
            params={"range": range_, "interval": interval},
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.debug("Chart fetch failed for %r: %s", symbol, e)
    return None


# ── action handlers ───────────────────────────────────────────────────────────

async def _action_quote(symbol: str, display_name: str, client: httpx.AsyncClient) -> str:
    """
    Fetch current quote with fundamentals.
    Pulls from the chart endpoint (1d range, 1d interval) which includes
    all the meta fields we need without a separate quoteSummary call.
    """
    data = await _fetch_chart(symbol, "1d", "1d", client)
    if data is None:
        return f"Error: Could not fetch quote for '{symbol}'."

    result_list = data.get("chart", {}).get("result", [])
    if not result_list:
        return f"Error: No data returned for '{symbol}'. The symbol may be delisted or invalid."

    meta     = result_list[0].get("meta", {})
    currency = meta.get("currency", "USD")
    exchange = meta.get("fullExchangeName") or meta.get("exchangeName", "")
    sym      = meta.get("symbol", symbol)
    name     = display_name or meta.get("longName") or meta.get("shortName") or sym

    price     = meta.get("regularMarketPrice")
    prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
    open_     = meta.get("regularMarketOpen")
    day_high  = meta.get("regularMarketDayHigh")
    day_low   = meta.get("regularMarketDayLow")
    volume    = meta.get("regularMarketVolume")

    if price is None:
        return f"Error: Price data unavailable for '{symbol}'."

    lines = [
        f"{name} ({sym}) — {exchange}",
        f"Currency: {currency}",
        "",
        f"Price:     {_fmt_price(price)} {currency}",
    ]

    # Change vs previous close
    if prev_close:
        change = price - prev_close
        pct    = (change / prev_close) * 100
        lines.append(f"Change:    {_fmt_change(change, pct)} vs prev close {_fmt_price(prev_close)}")

    # Pre/post market
    pre_price  = meta.get("preMarketPrice")
    post_price = meta.get("postMarketPrice")
    if pre_price:
        pre_chg = pre_price - price
        pre_pct = (pre_chg / price) * 100 if price else 0
        lines.append(f"Pre-market:  {_fmt_price(pre_price)} ({_fmt_change(pre_chg, pre_pct)})")
    if post_price:
        post_chg = post_price - price
        post_pct = (post_chg / price) * 100 if price else 0
        lines.append(f"After-hours: {_fmt_price(post_price)} ({_fmt_change(post_chg, post_pct)})")

    lines.append("")
    lines.append("Day's trading:")
    if open_:
        lines.append(f"  Open:    {_fmt_price(open_)} {currency}")
    if day_low is not None and day_high is not None:
        lines.append(f"  Range:   {_fmt_price(day_low)} – {_fmt_price(day_high)} {currency}")
    if volume:
        lines.append(f"  Volume:  {_fmt_large(volume)}")

    avg_vol = meta.get("averageDailyVolume3Month")
    if avg_vol:
        lines.append(f"  Avg vol (3mo): {_fmt_large(avg_vol)}")

    # 52-week range
    y52_high = meta.get("fiftyTwoWeekHigh")
    y52_low  = meta.get("fiftyTwoWeekLow")
    if y52_low is not None and y52_high is not None:
        lines.append(f"  52-week: {_fmt_price(y52_low)} – {_fmt_price(y52_high)} {currency}")

    # Moving averages
    ma50  = meta.get("fiftyDayAverage")
    ma200 = meta.get("twoHundredDayAverage")
    if ma50 or ma200:
        lines.append("")
        lines.append("Moving averages:")
        if ma50:
            diff50 = ((price - ma50) / ma50) * 100
            sign   = "+" if diff50 >= 0 else ""
            lines.append(f"  50-day MA:  {_fmt_price(ma50)} ({sign}{diff50:.1f}% vs current)")
        if ma200:
            diff200 = ((price - ma200) / ma200) * 100
            sign    = "+" if diff200 >= 0 else ""
            lines.append(f"  200-day MA: {_fmt_price(ma200)} ({sign}{diff200:.1f}% vs current)")

    # Fundamentals (stocks only — these fields are absent for crypto)
    mkt_cap  = meta.get("marketCap")
    pe_ratio = meta.get("trailingPE")
    div_yield = meta.get("dividendYield")

    fund_lines = []
    if mkt_cap:
        fund_lines.append(f"  Market cap:     {_fmt_large(mkt_cap)} {currency}")
    if pe_ratio:
        fund_lines.append(f"  P/E (trailing): {pe_ratio:.2f}")
    if div_yield:
        fund_lines.append(f"  Dividend yield: {div_yield * 100:.2f}%")

    if fund_lines:
        lines.append("")
        lines.append("Fundamentals:")
        lines.extend(fund_lines)

    return "\n".join(lines)


async def _action_history(
    symbol: str,
    display_name: str,
    period: str,
    client: httpx.AsyncClient,
) -> str:
    """
    Fetch OHLCV candle history for a given period.
    Interval is chosen automatically to keep the response size sensible.
    """
    if period not in _VALID_RANGES:
        valid = ", ".join(sorted(_VALID_RANGES))
        return f"Error: Invalid period '{period}'. Valid options: {valid}."

    interval = _RANGE_DEFAULTS[period]
    data = await _fetch_chart(symbol, period, interval, client)
    if data is None:
        return f"Error: Could not fetch history for '{symbol}'."

    result_list = data.get("chart", {}).get("result", [])
    if not result_list:
        return f"Error: No historical data found for '{symbol}'."

    res      = result_list[0]
    meta     = res.get("meta", {})
    currency = meta.get("currency", "USD")
    sym      = meta.get("symbol", symbol)
    name     = display_name or meta.get("longName") or meta.get("shortName") or sym

    timestamps = res.get("timestamp", [])
    quote_data = res.get("indicators", {}).get("quote", [{}])
    if not quote_data or not timestamps:
        return f"Error: Empty history data for '{symbol}'."

    q      = quote_data[0]
    opens  = q.get("open", [])
    highs  = q.get("high", [])
    lows   = q.get("low", [])
    closes = q.get("close", [])
    vols   = q.get("volume", [])

    # Build candle list, filtering None entries
    candles = []
    for i, ts in enumerate(timestamps):
        o = opens[i]  if i < len(opens)  else None
        h = highs[i]  if i < len(highs)  else None
        l = lows[i]   if i < len(lows)   else None
        c = closes[i] if i < len(closes) else None
        v = vols[i]   if i < len(vols)   else None
        if o is None or c is None:
            continue
        candles.append((ts, o, h, l, c, v))

    if not candles:
        return f"Error: No valid candle data for '{symbol}' over period '{period}'."

    # Limit output rows to keep prompt injection size reasonable
    max_rows   = 30
    show       = candles[-max_rows:]
    use_datetime = period == "1d"   # intraday → show time, else just date

    lines = [
        f"History: {name} ({sym}) | Period: {period} | Interval: {interval} | Currency: {currency}",
        "",
    ]

    if use_datetime:
        lines.append(f"{'Time (UTC)':<22} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>12}")
        lines.append("-" * 76)
        for ts, o, h, l, c, v in show:
            dt  = _ts_to_datetime(ts)
            vol = _fmt_large(v) if v else "N/A"
            lines.append(
                f"{dt:<22} {_fmt_price(o):>10} {_fmt_price(h):>10} "
                f"{_fmt_price(l):>10} {_fmt_price(c):>10} {vol:>12}"
            )
    else:
        lines.append(f"{'Date':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>12}")
        lines.append("-" * 66)
        for ts, o, h, l, c, v in show:
            date = _ts_to_date(ts)
            vol  = _fmt_large(v) if v else "N/A"
            lines.append(
                f"{date:<12} {_fmt_price(o):>10} {_fmt_price(h):>10} "
                f"{_fmt_price(l):>10} {_fmt_price(c):>10} {vol:>12}"
            )

    # Summary stats
    all_closes = [c for _, _, _, _, c, _ in candles]
    first_close = all_closes[0]
    last_close  = all_closes[-1]
    total_chg   = last_close - first_close
    total_pct   = (total_chg / first_close) * 100 if first_close else 0
    sign        = "+" if total_chg >= 0 else ""

    lines.append("")
    lines.append(
        f"Period return: {sign}{_fmt_price(total_chg)} ({sign}{total_pct:.2f}%) "
        f"| First close: {_fmt_price(first_close)} → Last close: {_fmt_price(last_close)}"
    )
    if len(candles) > max_rows:
        lines.append(f"(Showing last {max_rows} of {len(candles)} candles)")

    return "\n".join(lines)


async def _action_search(query: str, client: httpx.AsyncClient) -> str:
    """Search for a symbol by company name or keyword."""
    try:
        resp = await client.get(
            _YF_SEARCH_URL,
            params={"q": query, "quotesCount": 8, "newsCount": 0},
        )
        if resp.status_code != 200:
            return f"Error: Yahoo Finance search failed (HTTP {resp.status_code})."
        quotes = resp.json().get("quotes", [])
    except Exception as e:
        logger.warning("Finance search failed: %s", e)
        return f"Error: Search request failed — {e}."

    if not quotes:
        return f"No results found for '{query}'."

    lines = [f"Search results for '{query}':"]
    for q in quotes[:8]:
        sym   = q.get("symbol", "")
        name  = q.get("longname") or q.get("shortname") or ""
        type_ = q.get("quoteType", "")
        exch  = q.get("exchDisp") or q.get("exchange", "")
        lines.append(f"  {sym:<12} {name:<40} {type_:<10} {exch}")

    lines.append("")
    lines.append("Use the symbol from above with action='quote' or action='history'.")
    return "\n".join(lines)


# ── public async entrypoint ───────────────────────────────────────────────────

async def finance(
    query: str,
    action: str = "quote",
    period: str = "5d",
) -> str:
    """
    Retrieve financial data for stocks, ETFs, and cryptocurrencies.

    Args:
        query:   Ticker symbol (AAPL, RELIANCE.NS, BTC, ETH) or company name
                 ('Apple', 'Reliance Industries'). Company names are auto-resolved
                 to their ticker via Yahoo Finance search.
                 For Indian stocks, append .NS (NSE) or .BO (BSE): 'INFY.NS'.
                 For crypto, bare symbols like 'BTC' or 'ETH' work directly.
        action:  which data to return (see below). Default 'quote'.
        period:  history period for action='history'.
                 Valid: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max. Default '5d'.

    Actions:
        quote    — current price, change, day range, volume, 52-week range,
                   pre/post-market price, moving averages (50d/200d),
                   market cap, P/E ratio, dividend yield.
        history  — OHLCV candles for the given period. Interval is auto-selected
                   (e.g. daily candles for 1mo, weekly for 1y).
        search   — find a ticker symbol by company name or keyword.

    Returns a plain string in all cases. Never raises.
    """
    query = query.strip()
    if not query:
        return "Error: 'query' must not be empty."

    action = action.lower().strip()
    valid_actions = {"quote", "history", "search"}
    if action not in valid_actions:
        return f"Error: Unknown action '{action}'. Valid: {', '.join(sorted(valid_actions))}."

    try:
        async with httpx.AsyncClient(
            headers=_HEADERS,
            timeout=httpx.Timeout(12.0),
            follow_redirects=True,
        ) as client:
            if action == "search":
                return await _action_search(query, client)

            # Resolve symbol: apply crypto suffix, then try exact symbol first,
            # fall back to name search if the chart returns no data.
            symbol      = _resolve_symbol(query)
            display_name = ""

            # Always run search to get the display name, unless it looks like
            # an exact ticker (all caps, no spaces, short)
            looks_like_ticker = (
                query.upper() == query
                and " " not in query
                and len(query) <= 10
            )

            if not looks_like_ticker:
                # Company name — must resolve to a ticker first
                resolved, display_name = await _search_symbol(query, client)
                if resolved:
                    symbol = resolved

            if action == "quote":
                result = await _action_quote(symbol, display_name, client)
                # If exact ticker failed, try search-based resolution as fallback
                if result.startswith("Error:") and looks_like_ticker:
                    resolved, display_name = await _search_symbol(query, client)
                    if resolved and resolved != symbol:
                        symbol = resolved
                        result = await _action_quote(symbol, display_name, client)
                return result

            if action == "history":
                result = await _action_history(symbol, display_name, period, client)
                if result.startswith("Error:") and looks_like_ticker:
                    resolved, display_name = await _search_symbol(query, client)
                    if resolved and resolved != symbol:
                        symbol = resolved
                        result = await _action_history(symbol, display_name, period, client)
                return result

    except Exception as e:
        logger.exception("finance() top-level error for query=%r action=%r", query, action)
        return f"Error: Finance tool failed unexpectedly — {type(e).__name__}: {e}"

    return "Error: Unhandled action path."