"""
Finance tool — retrieves stock and cryptocurrency pricing, key statistics, and historical trends.
Uses Yahoo Finance keyless API endpoints.
"""

import logging
import datetime
import httpx

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

async def _resolve_ticker(query: str) -> tuple[str, str]:
    """
    Search Yahoo Finance autocomplete to resolve a name to a ticker symbol.
    Returns a tuple (symbol, long_name). If not found, returns (query, "").
    """
    search_url = "https://query2.finance.yahoo.com/v1/finance/search"
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=10.0) as client:
            resp = await client.get(search_url, params={"q": query})
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
    """
    Get current pricing, key stats, and optional 5-day history for a stock or cryptocurrency.
    
    Args:
        query: Ticker symbol (e.g., 'AAPL', 'BTC-USD') or company/asset name (e.g., 'Apple', 'Bitcoin').
        history: If True, includes daily open/close prices for the last 5 trading days.
        
    Returns:
        A formatted plain text summary of the financial data.
    """
    query = query.strip()
    if not query:
        return "[FINANCE ERROR: Empty query provided]"

    # Common crypto abbreviations to automatically suffix with -USD
    crypto_symbols = {
        "BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "DOT", "UNI", "LINK", "LTC", "BCH",
        "XLM", "ETC", "FIL", "VET", "TRX", "EOS", "XTZ", "AAVE", "MKR", "COMP", "YFI"
    }

    symbol = query
    if symbol.upper() in crypto_symbols:
        symbol = f"{symbol.upper()}-USD"

    asset_name = ""

    # First attempt: Try directly as a ticker symbol
    chart_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"interval": "1d", "range": "5d" if history else "1d"}
    
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=10.0) as client:
            resp = await client.get(chart_url, params=params)
            
            # If not direct ticker match, try resolving name
            if resp.status_code != 200:
                logger.info("Ticker '%s' not found directly. Searching for name resolution...", symbol)
                resolved_symbol, resolved_name = await _resolve_ticker(query)
                if resolved_symbol.upper() != symbol.upper():
                    symbol = resolved_symbol
                    asset_name = resolved_name
                    chart_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                    resp = await client.get(chart_url, params=params)
                    
            if resp.status_code != 200:
                return f"[FINANCE ERROR: Could not resolve financial symbol or asset for '{query}']"
                
            chart_data = resp.json()
    except Exception as e:
        logger.exception("Yahoo Finance API request failed for: %s", query)
        return f"[FINANCE ERROR: Yahoo Finance API failed — {type(e).__name__}: {str(e)}]"

    result_list = chart_data.get("chart", {}).get("result", [])
    if not result_list:
        return f"[FINANCE ERROR: No data found for symbol '{symbol}']"

    res = result_list[0]
    meta = res.get("meta", {})
    
    # Pricing info
    current_price = meta.get("regularMarketPrice")
    prev_close = meta.get("chartPreviousClose")
    currency = meta.get("currency", "USD")
    exchange = meta.get("fullExchangeName") or meta.get("exchangeName", "Unknown")
    symbol = meta.get("symbol", symbol)
    long_name = asset_name or meta.get("longName") or meta.get("shortName") or symbol

    if current_price is None:
        return f"[FINANCE ERROR: Could not retrieve current price for symbol '{symbol}']"

    # Helper formatting function for prices
    def fmt_price(val) -> str:
        if val is None:
            return "N/A"
        if val < 0.1:
            return f"{val:.6f}"
        if val < 2.0:
            return f"{val:.4f}"
        return f"{val:.2f}"

    lines = [
        f"Financial Data for: {long_name} ({symbol})",
        f"Exchange: {exchange} | Currency: {currency}",
        f"Current Price: {fmt_price(current_price)} {currency}"
    ]

    # Price change
    if prev_close is not None:
        change = current_price - prev_close
        pct_change = (change / prev_close) * 100 if prev_close != 0 else 0.0
        sign = "+" if change >= 0 else ""
        lines.append(f"Price Change: {sign}{fmt_price(change)} ({sign}{pct_change:.2f}%) vs Previous Close ({fmt_price(prev_close)} {currency})")

    # Additional stats
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

    # History (last 5 trading days)
    if history:
        timestamps = res.get("timestamp", [])
        quote = res.get("indicators", {}).get("quote", [{}])[0]
        opens = quote.get("open", [])
        closes = quote.get("close", [])

        if timestamps and opens and closes:
            lines.append("")
            lines.append("5-Day Trading History:")
            # Zip and show last 5 days
            history_data = list(zip(timestamps, opens, closes))
            # Filter out entries where open or close is None
            history_data = [(t, o, c) for t, o, c in history_data if o is not None and c is not None]
            
            for t, o, c in history_data[-5:]:
                try:
                    date_str = datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).strftime('%Y-%m-%d')
                except Exception:
                    date_str = "N/A"
                lines.append(f"  - {date_str}: Open {fmt_price(o)} | Close {fmt_price(c)} {currency}")

    return "\n".join(lines)
