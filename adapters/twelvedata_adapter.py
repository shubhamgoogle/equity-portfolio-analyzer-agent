"""Twelve Data adapter — third-party fundamentals & valuation source.

Twelve Data is a financial-data API with global coverage (stocks, ETFs,
forex, crypto). It complements yfinance with cleaner valuation ratios
and a polished company profile, and gives us a useful cross-check on
the live quote.

Endpoints used per lookup (3 calls total):
  - `td.quote(symbol)`           → live price, day range, 52-week range
                                   (free / Basic tier, 1 credit)
  - `td.get_statistics(symbol)`  → P/E, market cap, dividend yield, beta
                                   (Pro tier, 50 credits)
  - `td.get_profile(symbol)`     → sector, industry, description, employees
                                   (Grow tier, 10 credits)

NOTE on plan tiers: per Twelve Data's current pricing, only `/quote`
is on the free Basic plan. `/profile` requires Grow and `/statistics`
requires Pro. If running on the Basic plan, expect those two calls to
fail and only quote-derived fields to be populated. None of the other
fundamentals endpoints offered by the SDK (earnings, dividends,
income_statement, balance_sheet, cash_flow, insider_transactions,
institutional_holders, fund_holders) are available on the free tier,
which is why we don't wire them in.

Each call is wrapped independently so a failure in one (e.g. profile not
available for an ETF, or the symbol's plan tier blocking access) does
not blank the others.

TTL cache: results are memoized per symbol for `TWELVE_DATA_CACHE_TTL`
seconds (default 300s = 5 min) to avoid burning API credits when the
same ticker is queried repeatedly in a short window. Set the env var
to 0 to disable caching.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from twelvedata import TDClient

from adapters.base_adapter import BaseStockAdapter

logger = logging.getLogger(__name__)


def _ttl_from_env() -> float:
    raw = os.getenv("TWELVE_DATA_CACHE_TTL")
    if raw is None or raw.strip() == "":
        return 300.0
    try:
        val = float(raw)
        return max(0.0, val)
    except ValueError:
        logger.warning(
            "Invalid TWELVE_DATA_CACHE_TTL=%r; falling back to 300s.", raw
        )
        return 300.0


class TwelveDataAdapter(BaseStockAdapter):
    """Aggregates quote + statistics + profile from Twelve Data."""

    def __init__(self) -> None:
        api_key = os.getenv("TWELVE_API_KEY")
        if not api_key:
            self._client: TDClient | None = None
            self._init_error: str | None = (
                "TWELVE_API_KEY is not set. Add it to your .env file."
            )
        else:
            self._client = TDClient(apikey=api_key)
            self._init_error = None

        self._cache_ttl: float = _ttl_from_env()
        self._cache: dict[str, tuple[float, dict]] = {}
        self._cache_lock = threading.Lock()

    def get_stock_info(self, stock_name: str) -> dict:
        if self._client is None:
            return {"error": self._init_error or "Twelve Data client not initialized."}

        symbol = (stock_name or "").strip()
        if not symbol:
            return {"error": "Empty symbol passed to TwelveDataAdapter."}

        cache_key = symbol.upper()
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        out: dict[str, Any] = {
            "source": "Twelve Data",
            "symbol": cache_key,
        }
        errors: list[str] = []

        out.update(self._fetch_quote(symbol, errors))
        out.update(self._fetch_statistics(symbol, errors))
        out.update(self._fetch_profile(symbol, errors))

        # If literally every Twelve Data call failed, surface a clean error.
        useful = {
            k: v
            for k, v in out.items()
            if k not in ("source", "symbol") and v is not None
        }
        if not useful:
            # Don't cache outright failures — the symbol may have been a
            # typo the user corrects on the next turn.
            return {
                "error": f"Twelve Data returned no usable fields for '{symbol}'.",
                "details": errors,
            }
        if errors:
            out["partial_errors"] = errors

        self._cache_put(cache_key, out)
        return out

    # --- TTL cache ---------------------------------------------------------

    def _cache_get(self, key: str) -> dict | None:
        if self._cache_ttl <= 0:
            return None
        now = time.time()
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            ts, payload = entry
            if now - ts >= self._cache_ttl:
                # Stale → evict so the dict doesn't grow unbounded.
                self._cache.pop(key, None)
                return None
            # Shallow copy + mark as cache hit so consumers can tell
            # whether the data is fresh or memoized.
            cloned = dict(payload)
            cloned["_cached"] = True
            cloned["_cache_age_seconds"] = round(now - ts, 1)
            return cloned

    def _cache_put(self, key: str, value: dict) -> None:
        if self._cache_ttl <= 0:
            return
        with self._cache_lock:
            self._cache[key] = (time.time(), dict(value))

    # --- Quote -------------------------------------------------------------

    def _fetch_quote(self, symbol: str, errors: list[str]) -> dict:
        try:
            raw = self._client.quote(symbol=symbol).as_json()
        except Exception as e:  # noqa: BLE001
            errors.append(f"quote: {e}")
            return {}
        if not isinstance(raw, dict):
            errors.append(f"quote: unexpected shape {type(raw).__name__}")
            return {}
        if raw.get("status") == "error" or "code" in raw and "message" in raw:
            errors.append(f"quote: {raw.get('message', raw)}")
            return {}

        fw = raw.get("fifty_two_week") or {}
        return {
            "companyName": raw.get("name"),
            "exchange": raw.get("exchange"),
            "micCode": raw.get("mic_code"),
            "currency": raw.get("currency"),
            "asOf": raw.get("datetime"),
            "isMarketOpen": raw.get("is_market_open"),
            "open": raw.get("open"),
            "previousClose": raw.get("previous_close"),
            "lastPrice": raw.get("close"),
            "dayHigh": raw.get("high"),
            "dayLow": raw.get("low"),
            "change": raw.get("change"),
            "pChange": raw.get("percent_change"),
            "volume": raw.get("volume"),
            "averageVolume": raw.get("average_volume"),
            "fiftyTwoWeekHigh": fw.get("high"),
            "fiftyTwoWeekLow": fw.get("low"),
            "fiftyTwoWeekRange": fw.get("range"),
        }

    # --- Statistics --------------------------------------------------------

    def _fetch_statistics(self, symbol: str, errors: list[str]) -> dict:
        try:
            raw = self._client.get_statistics(symbol=symbol).as_json()
        except Exception as e:  # noqa: BLE001
            errors.append(f"statistics: {e}")
            return {}
        if not isinstance(raw, dict):
            errors.append(f"statistics: unexpected shape {type(raw).__name__}")
            return {}

        # The live API returns the metric blocks at the top level (no
        # outer `statistics` wrapper). Probed empirically on 2026-05-12.
        val = raw.get("valuations_metrics") or {}
        fin = raw.get("financials") or {}
        price = raw.get("stock_price_summary") or {}
        div = raw.get("dividends_and_splits") or {}

        return {
            "marketCap": val.get("market_capitalization"),
            "enterpriseValue": val.get("enterprise_value"),
            "trailingPE": val.get("trailing_pe"),
            "forwardPE": val.get("forward_pe"),
            "pegRatio": val.get("peg_ratio"),
            "priceToSales": val.get("price_to_sales_ttm"),
            "priceToBook": val.get("price_to_book_mrq"),
            "profitMargin": fin.get("profit_margin"),
            "operatingMargin": fin.get("operating_margin"),
            "returnOnEquity": fin.get("return_on_equity_ttm"),
            "returnOnAssets": fin.get("return_on_assets_ttm"),
            "beta": price.get("beta"),
            "fiftyTwoWeekChangePct": price.get("fifty_two_week_change"),
            "fiftyDayMA": price.get("day_50_ma"),
            "twoHundredDayMA": price.get("day_200_ma"),
            "dividendYield": div.get("forward_annual_dividend_yield"),
            "dividendRate": div.get("forward_annual_dividend_rate"),
            "payoutRatio": div.get("payout_ratio"),
        }

    # --- Profile -----------------------------------------------------------

    def _fetch_profile(self, symbol: str, errors: list[str]) -> dict:
        try:
            raw = self._client.get_profile(symbol=symbol).as_json()
        except Exception as e:  # noqa: BLE001
            errors.append(f"profile: {e}")
            return {}
        if not isinstance(raw, dict):
            errors.append(f"profile: unexpected shape {type(raw).__name__}")
            return {}

        description = raw.get("description")
        if isinstance(description, str) and len(description) > 600:
            # Trim to keep token usage sane — the LLM only needs a snippet.
            description = description[:600].rsplit(" ", 1)[0] + "…"

        return {
            "sector": raw.get("sector"),
            "industry": raw.get("industry"),
            "country": raw.get("country"),
            "employees": raw.get("employees"),
            "website": raw.get("website"),
            "ceo": raw.get("CEO") or raw.get("ceo"),
            "description": description,
        }
