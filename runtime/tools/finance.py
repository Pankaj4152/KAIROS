"""
Finance tool — retrieves stock and cryptocurrency pricing, key statistics, and historical trends.
Uses Yahoo Finance keyless API endpoints. Optimized for KAIROS.
"""

import logging
import datetime
import httpx

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/json,application/xhtml+xml,application/xml;q=0.9",
    "Accept-Language": "en-US,en;q=0.9"
}

# Common crypto abbreviations to automatically suffix with -USD
CRYPTO_SYMBOLS = {
    "BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "DOT", "UNI", "LINK", "LTC", "BCH",
    "XLM", "ETC", "FIL", "VET", "TRX", "EOS", "XTZ", "AAVE", "MKR", "COMP", "YFI"
}

async def _resolve_ticker(query: str) -> tuple[str, str]:
    search_url = "https://query2.finance.yahoo.com/v1/finance/search"
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(search_url, params={"q": query, "quotesCount": 1})
            if resp.status_code == 200:
                data = resp.json()
                quotes = data.get("quotes", [])
                if quotes:
                    best_match = quotes[0]
                    symbol = best_match.get("symbol", query)
                    name = best_match.get("longname") or best_match.get("shortname") or symbol
                    return symbol, name
    except Exception as e:
        logger.warning("Failed to resolve ticker for '%s': %s", query, e)
    return query, ""

async def finance(query: str, history: bool = False) -> str:
    query = query.strip()
    if not query:
        return "[FINANCE ERROR: Empty query provided]"

    symbol = query
    if symbol.upper() in CRYPTO_SYMBOLS:
        symbol = f"{symbol.upper()}-USD"

    asset_name = ""
    chart_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"interval": "1d", "range": "5d" if history else "1d"}
    
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(chart_url, params=params)
            
            if resp.status_code != 200:
                resolved_symbol, resolved_name = await _resolve_ticker(query)
                if resolved_symbol.upper() != symbol.upper():
                    symbol = resolved_symbol
                    asset_name = resolved_name
                    chart_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                    resp = await client.get(chart_url, params=params)
                    
            if resp.status_code != 200:
                return f"[FINANCE ERROR: Could not resolve financial asset for '{query}']"
                
            chart_data = resp.json()
    except Exception as e:
        logger.exception("Yahoo Finance API request failed for: %s", query)
        return f"[FINANCE ERROR: Yahoo Finance API failed — {type(e).__name__}: {str(e)}]"

    result_list = chart_data.get("chart", {}).get("result", [])
    if not result_list:
        return f"[FINANCE ERROR: No data found for symbol '{symbol}']"

    res = result_list[0]
    meta = res.get("meta", {})
    
    current_price = meta.get("regularMarketPrice")
    prev_close = meta.get("chartPreviousClose")
    currency = meta.get("currency", "USD")
    exchange = meta.get("fullExchangeName") or meta.get("exchangeName", "Unknown")
    symbol = meta.get("symbol", symbol)
    long_name = asset_name or meta.get("longName") or meta.get("shortName") or symbol

    if current_price is None:
        return f"[FINANCE ERROR: Could not retrieve current price for symbol '{symbol}']"

    def fmt_price(val) -> str:
        if val is None: return "N/A"
        if val < 0.1: return f"{val:.6f}"
        if val < 2.0: return f"{val:.4f}"
        return f"{val:.2f}"

    lines = [
        f"Financial Data for: {long_name} ({symbol})",
        f"Exchange: {exchange} | Currency: {currency}",
        f"Current Price: {fmt_price(current_price)} {currency}"
    ]

    if prev_close is not None:
        change = current_price - prev_close
        pct_change = (change / prev_close) * 100 if prev_close != 0 else 0.0
        sign = "+" if change >= 0 else ""
        lines.append(f"Price Change: {sign}{fmt_price(change)} ({sign}{pct_change:.2f}%) vs Previous Close ({fmt_price(prev_close)} {currency})")

    day_high = meta.get("regularMarketDayHigh")
    day_low = meta.get("regularMarketDayLow")
    volume = meta.get("regularMarketVolume")
    y52_high = meta.get("fiftyTwoWeekHigh")
    y52_low = meta.get("fiftyTwoWeekLow")

    lines.append("Key Statistics:")
    if day_low is not None and day_high is not None:
        lines.append(f"  - Day's Range: {fmt_price(day_low)} - {fmt_price(day_high)} {currency}")
    if volume is not None:
        lines.append(f"  - Day's Volume: {volume:,}")
    if y52_low is not None and y52_high is not None:
        lines.append(f"  - 52-Week Range: {fmt_price(y52_low)} - {fmt_price(y52_high)} {currency}")

    if history:
        timestamps = res.get("timestamp", [])
        indicators = res.get("indicators", {}).get("quote", [{}])
        
        if indicators and timestamps:
            quote = indicators[0]
            opens = quote.get("open", [])
            closes = quote.get("close", [])
            
            history_data = list(zip(timestamps, opens, closes))
            # Filter systematically before array index slice allocations
            history_data = [(t, o, c) for t, o, c in history_data if o is not None and c is not None]
            
            if history_data:
                lines.append("\n5-Day Trading History:")
                for t, o, c in history_data[-5:]:
                    try:
                        date_str = datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).strftime('%Y-%m-%d')
                    except Exception:
                        date_str = "N/A"
                    lines.append(f"  - {date_str}: Open {fmt_price(o)} | Close {fmt_price(c)} {currency}")

    return "\n".join(lines)
