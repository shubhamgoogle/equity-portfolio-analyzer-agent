"""ADK-compatible tools that wrap the `adapters/` data-source layer.

Architecture:
    Instead of exposing one tool per adapter (which forces the LLM into
    a multi-turn loop and lets it forget to call one), we expose a
    SINGLE aggregator tool `analyze_security`. It fans out to every
    adapter concurrently, then hands the LLM one combined payload so
    the model can produce a synthesized verdict in a single pass.

Adding a new data source:
    1. Implement an adapter under `adapters/` that satisfies
       `BaseStockAdapter.get_stock_info(name) -> dict`.
    2. Add it to the fan-out list inside `analyze_security` below.
    3. Mention the new source's key in the agent's instruction so the
       LLM knows what to look for in the combined payload.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from typing import Any

from adapters.nse_bse_adapter import NseBseAdapter
from adapters.tavily_adapter import TavilyAdapter
from adapters.twelvedata_adapter import TwelveDataAdapter
from adapters.yfinance_adapter import YFinanceAdapter

# Adapter instances are constructed once and reused across calls so that
# any internal caches (HTTP sessions, BSE scrip-code table, etc.) survive.
_yfinance_adapter = YFinanceAdapter()
_twelvedata_adapter = TwelveDataAdapter()
_nse_bse_adapter = NseBseAdapter()
_tavily_adapter = TavilyAdapter()

# Per-adapter timeout (seconds). The NSE scraper occasionally takes a
# while warming its cookie session, and TwelveData now serializes 4
# indicator calls + 3 fundamentals calls = up to 7 HTTP round-trips
# inside the 8 req/min Basic-tier window. Keep this generous so a slow
# warm-up doesn't blank the technicals block; the cache amortizes most
# of the cost across repeated lookups.
_ADAPTER_TIMEOUT = 45.0


def _to_jsonable(value: Any) -> Any:
    """Coerce arbitrary values (numpy scalars, datetimes, ...) into JSON-safe types."""
    return json.loads(json.dumps(value, default=str))


def _looks_indian(symbol: str) -> bool:
    """Best-effort heuristic for whether NSE/BSE should be consulted.

    True for:
      - bare alpha tickers that look NSE-style (e.g. RELIANCE, TCS) — we
        can't tell for sure from the symbol alone, so we err on the side
        of calling NSE/BSE and letting them return a clean error if it
        isn't actually listed there.
      - all-digit input (BSE scrip code, e.g. 500325).
      - Yahoo-style Indian suffixes (.NS / .BO).

    False only for tickers that *clearly* aren't Indian — i.e. anything
    with a non-Indian Yahoo suffix (.L, .TO, .HK, ...). For typical US
    tickers like AAPL we still try NSE/BSE; the call is fast and the
    error path is harmless.
    """
    s = symbol.strip().upper()
    if not s:
        return False
    if s.isdigit():
        return True
    if s.endswith(".NS") or s.endswith(".BO"):
        return True
    # Any non-Indian Yahoo exchange suffix → skip Indian sources.
    if "." in s:
        suffix = s.rsplit(".", 1)[-1]
        if suffix not in ("NS", "BO"):
            return False
    return True


def _normalize_for_nse(symbol: str) -> str:
    """Strip Yahoo-style Indian suffixes so NSE/BSE adapter gets the bare symbol."""
    s = symbol.strip().upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        s = s.rsplit(".", 1)[0]
    return s


def _normalize_for_twelvedata(symbol: str) -> str:
    """Twelve Data doesn't accept Yahoo's `.NS` / `.BO` Indian suffixes —
    it expects the bare ticker (and resolves the exchange itself). Strip
    those suffixes; leave other Yahoo suffixes (.L, .TO, etc.) alone
    since Twelve Data handles them and we don't want to over-strip."""
    s = symbol.strip()
    upper = s.upper()
    if upper.endswith(".NS") or upper.endswith(".BO"):
        return s.rsplit(".", 1)[0]
    return s


def _safe_call(adapter_name: str, fn, *args) -> dict:
    """Run an adapter call, returning either its dict result or an error dict."""
    try:
        result = fn(*args)
        if not isinstance(result, dict):
            return {"error": f"{adapter_name} returned non-dict: {type(result).__name__}"}
        return result
    except Exception as e:  # noqa: BLE001
        return {"error": f"{adapter_name} raised: {e}"}


def analyze_security(symbol: str) -> dict:
    """Fan out to every data source for a security and return a combined payload.

    This is the ONLY tool the agent calls. It runs all four adapters in
    parallel and returns their normalized responses in a single dict, so
    the LLM gets the full picture in one shot and can synthesize a
    verdict without making multiple tool round-trips.

    Sources covered (in parallel):
        - Yahoo Finance  → fundamentals (P/E, market cap, sector,
                           dividend yield, 52w range, long business
                           summary). Best for US/global equities.
        - Twelve Data    → independent fundamentals + valuation ratios
                           + clean company profile. Cross-checks Yahoo.
        - NSE / BSE      → direct exchange quote for Indian-listed
                           equities (live INR price, day range,
                           exchange-reported industry). Auto-skipped
                           for clearly non-Indian tickers.
        - Tavily         → recent news, analyst commentary, regulatory
                           events, and overall sentiment.

    Args:
        symbol: Ticker symbol or company name. Examples:
            - US:       "AAPL", "MSFT", "VFIAX", "SPY"
            - Indian:   "RELIANCE", "TCS", "INFY", "HDFCBANK"
            - Indian (Yahoo style):  "RELIANCE.NS", "TCS.BO"
            - BSE code: "500325" (Reliance), "532540" (TCS)
            - Free-form: "Reliance Industries"
          The adapter rewrites the symbol per-source as needed (e.g.
          strips ".NS"/".BO" before sending to the NSE/BSE adapter).

    Returns:
        A dict with the following keys:
            - input_symbol:  the symbol the user provided
            - yahoo:         result from Yahoo Finance, or {"error": ...}
            - twelvedata:    result from Twelve Data, or {"error": ...}
            - nse_bse:       result from NSE/BSE (or {"skipped": "..."} if
                             the heuristic decided the symbol is clearly
                             non-Indian), or {"error": ...}
            - news:          result from Tavily news search
            - elapsed_ms:    how long the fan-out took (debugging aid)
            - all_failed:    True if every source returned an error; the
                             LLM should ask the user for a corrected
                             symbol in that case.
    """
    started = time.perf_counter()
    raw_symbol = (symbol or "").strip()
    upper = raw_symbol.upper()
    nse_symbol = _normalize_for_nse(raw_symbol)
    td_symbol = _normalize_for_twelvedata(raw_symbol)
    indian = _looks_indian(raw_symbol)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        fut_yahoo = pool.submit(
            _safe_call, "yfinance", _yfinance_adapter.get_stock_info, raw_symbol
        )
        fut_twelve = pool.submit(
            _safe_call, "twelvedata", _twelvedata_adapter.get_stock_info, td_symbol
        )
        if indian:
            fut_nse = pool.submit(
                _safe_call, "nse_bse", _nse_bse_adapter.get_stock_info, nse_symbol
            )
        else:
            fut_nse = None
        fut_news = pool.submit(
            _safe_call, "tavily", _tavily_adapter.get_stock_info, raw_symbol
        )

        def _resolve(fut):
            if fut is None:
                return None
            try:
                return fut.result(timeout=_ADAPTER_TIMEOUT)
            except concurrent.futures.TimeoutError:
                return {"error": "adapter timed out"}
            except Exception as e:  # noqa: BLE001
                return {"error": f"adapter raised: {e}"}

        yahoo = _resolve(fut_yahoo)
        twelve = _resolve(fut_twelve)
        nse = _resolve(fut_nse)
        news = _resolve(fut_news)

    if nse is None:
        nse = {"skipped": "Symbol does not look NSE/BSE listed; Indian-exchange call skipped."}

    sources = (yahoo, twelve, nse, news)
    all_failed = all(
        isinstance(s, dict) and "error" in s and "skipped" not in s
        for s in sources
    )

    payload = {
        "input_symbol": raw_symbol,
        "normalized": {
            "upper": upper,
            "nse_symbol": nse_symbol,
            "td_symbol": td_symbol,
            "looks_indian": indian,
        },
        "yahoo": yahoo,
        "twelvedata": twelve,
        "nse_bse": nse,
        "news": news,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "all_failed": all_failed,
    }
    return _to_jsonable(payload)


ALL_TOOLS = [analyze_security]
