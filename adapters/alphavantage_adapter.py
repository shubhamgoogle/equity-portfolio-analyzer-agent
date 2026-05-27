"""Alpha Vantage adapter — retrieves comprehensive stock, ETF, and fundamental market data.

Integrates directly with the locally deployed Alpha Vantage MCP codebase, loading
modules dynamically without needing a separate running MCP network process.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Any

# Add local Alpha Vantage packages to path
local_api_path = "/Users/shubu/Documents/github_repo/equity-portfolio-analyzer-agent/mcps/local/api/src"
local_mcp_path = "/Users/shubu/Documents/github_repo/equity-portfolio-analyzer-agent/mcps/local/mcp/src"
if local_api_path not in sys.path:
    sys.path.insert(0, local_api_path)
if local_mcp_path not in sys.path:
    sys.path.insert(0, local_mcp_path)

from av_api.context import set_api_key
from av_api.registry import call_tool

from adapters.base_adapter import BaseStockAdapter

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Exception raised when Alpha Vantage API rate limit is reached."""
    pass


def _ttl_from_env() -> float:
    raw = os.getenv("ALPHAVANTAGE_CACHE_TTL")
    if raw is None or raw.strip() == "":
        return 300.0  # 5 min default
    try:
        val = float(raw)
        return max(0.0, val)
    except ValueError:
        return 300.0


class AlphaVantageAdapter(BaseStockAdapter):
    """Aggregates Quote + Overview + Statement / ETF details from local Alpha Vantage modules."""

    def __init__(self) -> None:
        self._cache_ttl: float = _ttl_from_env()
        self._cache: dict[str, tuple[float, dict]] = {}
        self._cache_lock = threading.Lock()

    def get_stock_info(self, stock_name: str) -> dict:
        api_key = os.getenv("ALPHAVANTAGE_API")
        if not api_key:
            return {"error": "ALPHAVANTAGE_API is not set in the environment. Please check your .env file."}

        set_api_key(api_key)

        symbol = (stock_name or "").strip().upper()
        if not symbol:
            return {"error": "Empty symbol passed to AlphaVantageAdapter."}

        cached = self._cache_get(symbol)
        if cached is not None:
            return cached

        out: dict[str, Any] = {
            "source": "Alpha Vantage Local",
            "symbol": symbol,
        }
        errors: list[str] = []

        try:
            # 1. Fetch Quote
            quote_data = self._fetch_quote(symbol)
            if quote_data:
                out.update(quote_data)

            # Sleep 1.2s to avoid hitting 1 request/sec free tier rate limits
            time.sleep(1.2)

            # 2. Fetch Company Overview
            overview = self._fetch_overview(symbol)
            is_etf = False
            if overview:
                if not overview.get("Name") or "assettype" in [k.lower() for k in overview.keys()] and "etf" in str(overview.get("AssetType", "")).lower():
                    is_etf = True
                else:
                    out["overview"] = overview
            else:
                is_etf = True

            if is_etf:
                # Sleep 1.2s before querying ETF Profile
                time.sleep(1.2)
                etf = self._fetch_etf_profile(symbol)
                if etf:
                    out["etf_profile"] = etf

        except RateLimitError as e:
            errors.append(str(e))
            out["rate_limited"] = True
        except Exception as e:
            errors.append(f"Alpha Vantage error: {e}")

        # Check if we got anything useful
        useful = {
            k: v
            for k, v in out.items()
            if k not in ("source", "symbol") and v is not None
        }
        if not useful:
            return {
                "error": f"Alpha Vantage returned no usable fields for '{symbol}'.",
                "details": errors,
            }

        if errors:
            out["partial_errors"] = errors

        # Only cache if we weren't rate limited so we can try again later
        if not out.get("rate_limited"):
            self._cache_put(symbol, out)

        return out

    def _check_rate_limit(self, res: Any) -> None:
        if isinstance(res, dict):
            # Alpha Vantage rate limits contain keys like 'Information' or 'Note'
            for k in ("Information", "Note", "information", "note"):
                if k in res:
                    msg = res[k]
                    raise RateLimitError(f"Alpha Vantage rate limit: {msg}")
        elif isinstance(res, str):
            if "thank you for using alpha vantage" in res.lower() or "please consider spreading out your free api requests" in res.lower():
                raise RateLimitError("Alpha Vantage rate limit reached (thank you for using Alpha Vantage msg)")

    def _fetch_quote(self, symbol: str) -> dict | None:
        res = call_tool("GLOBAL_QUOTE", {"symbol": symbol})
        self._check_rate_limit(res)
        
        if isinstance(res, str):
            if "symbol,open" in res.lower():
                lines = res.strip().split("\n")
                if len(lines) > 1:
                    headers = lines[0].strip().split(",")
                    values = lines[1].strip().split(",")
                    return dict(zip(headers, values))
        elif isinstance(res, dict):
            return res.get("Global Quote") or res
        return None

    def _fetch_overview(self, symbol: str) -> dict | None:
        res = call_tool("COMPANY_OVERVIEW", {"symbol": symbol})
        self._check_rate_limit(res)
        
        if isinstance(res, dict):
            return res
        elif isinstance(res, str):
            import json
            try:
                parsed = json.loads(res)
                self._check_rate_limit(parsed)
                return parsed
            except json.JSONDecodeError:
                return {"raw_text": res}
        return None

    def _fetch_etf_profile(self, symbol: str) -> dict | None:
        res = call_tool("ETF_PROFILE", {"symbol": symbol})
        self._check_rate_limit(res)
        
        if isinstance(res, dict):
            return res
        elif isinstance(res, str):
            import json
            try:
                parsed = json.loads(res)
                self._check_rate_limit(parsed)
                return parsed
            except json.JSONDecodeError:
                return {"raw_text": res}
        return None

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
                self._cache.pop(key, None)
                return None
            cloned = dict(payload)
            cloned["_cached"] = True
            cloned["_cache_age_seconds"] = round(now - ts, 1)
            return cloned

    def _cache_put(self, key: str, value: dict) -> None:
        if self._cache_ttl <= 0:
            return
        with self._cache_lock:
            self._cache[key] = (time.time(), dict(value))
