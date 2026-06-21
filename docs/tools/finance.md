# finance

**← [Back to Tool Guide](../TOOL_GUIDE.md)**

**File:** `runtime/tools/finance.py`
**Env vars required:** none — Yahoo Finance public endpoints are keyless
**Execution pattern:** agentic
**Added:** Mark 3

Stock, ETF, and cryptocurrency pricing, fundamentals, and historical data.

## When it's used

- User asks for a stock/crypto price, market cap, or trading history
- User wants to resolve a company name to a ticker symbol
- Classifier sets `domains: [finance]` and `tools_needed: [finance]`

## Parameters

| Parameter | Type   | Required | Constraints                                                    | Description |
|------------|--------|----------|-------------------------------------------------------------------|--------------|
| `query`    | string | yes      | 1–100 chars                                                        | Ticker or company name — see symbol formats below |
| `action`   | string | no       | enum: `quote` (default), `history`, `search`                       | Which data to retrieve |
| `period`   | string | no       | enum: `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `max` (default `5d`) | History window for `action='history'` |

**Symbol formats:**
- US stocks: bare symbol — `AAPL`, `TSLA`, `SPY`
- Indian NSE: append `.NS` — `INFY.NS`, `RELIANCE.NS`
- Indian BSE: append `.BO`
- Crypto: bare symbol, auto-converted — `BTC` → `BTC-USD`, `ETH` → `ETH-USD`
- Company names — `Apple`, `Reliance Industries` — auto-resolved to a ticker via Yahoo Finance search

## Actions

| Action     | What it does |
|-------------|---------------|
| `quote`     | Current price, change vs previous close, day's range, volume (+ 3-month average), 52-week range, 50/200-day moving averages, pre/post-market price when available, market cap, trailing P/E, dividend yield (stocks only — these fields are absent for crypto). **Default.** |
| `history`   | OHLCV candles for `period`. Interval auto-selected per period (`1d`→5m, `5d`/`1mo`/`3mo`→1d, `6mo`/`1y`→1wk, `2y`/`5y`/`max`→1mo) to keep output concise. Output capped at the last 30 rows; total candle count is reported if more exist. |
| `search`    | Resolve a company name or partial ticker to up to 8 matching symbols, with name, asset type, and exchange. |

## Example LLM calls

**Quote (default action):**
```json
{ "name": "finance", "input": { "query": "AAPL" } }
```

**Crypto quote:**
```json
{ "name": "finance", "input": { "query": "BTC" } }
```

**History:**
```json
{ "name": "finance", "input": { "query": "TSLA", "action": "history", "period": "1mo" } }
```

**Resolve a company name to a ticker:**
```json
{ "name": "finance", "input": { "query": "Reliance Industries", "action": "search" } }
```

## Return format (quote)

```
Apple Inc. (AAPL) — NasdaqGS
Currency: USD

Price:     189.50 USD
Change:    +2.30 (+1.23%) vs prev close 187.20

Day's trading:
  Open:    188.50 USD
  Range:   186.50 – 191.00 USD
  Volume:  45.00M
  Avg vol (3mo): 40.00M
  52-week: 150.00 – 230.00 USD

Moving averages:
  50-day MA:  182.00 (+4.1% vs current)
  200-day MA: 175.00 (+8.3% vs current)

Fundamentals:
  Market cap:     2.90T USD
  P/E (trailing): 29.50
  Dividend yield: 0.50%
```

## Return format (history)

```
History: Apple Inc. (AAPL) | Period: 1mo | Interval: 1d | Currency: USD

Date         Open       High        Low      Close      Volume
------------------------------------------------------------------
2026-05-21   185.20    187.00     184.50    186.10      42.10M
...

Period return: +5.30 (+2.85%) | First close: 186.10 → Last close: 191.40
```

## Failure modes

| Condition                                  | Return value |
|-----------------------------------------------|-----------------|
| Empty query                                    | `Error: 'query' must not be empty.` |
| Unknown action                                  | `Error: Unknown action '<action>'. Valid: history, quote, search.` |
| Invalid `period` for `history`                    | `Error: Invalid period '<period>'. Valid options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max.` |
| Symbol not found / delisted                        | `Error: No data returned for '<symbol>'. The symbol may be delisted or invalid.` |
| No price data in response                            | `Error: Price data unavailable for '<symbol>'.` |
| No historical candles for the period                   | `Error: No valid candle data for '<symbol>' over period '<period>'.` |
| Search returns nothing                                    | `No results found for '<query>'.` |
| Network/HTTP failure                                        | `Error: Finance tool failed unexpectedly — <ExceptionType>: <detail>` |

## Notes

- Exact-looking tickers (`query.upper() == query`, no spaces, ≤10 chars) skip the symbol-search round-trip and try the chart endpoint directly first; only falls back to search if that fails. Company names always search first.
- All large numbers are formatted for readability: `2.90T`, `456.70B`, `1.50M`.
