"""Combined NSE + BSE adapter for live Indian equity quotes.

`yfinance` works for Indian tickers via `.NS` / `.BO` suffixes but is often
stale, missing fields, or just returns empty dicts. This adapter goes
straight to the source by wrapping three community libraries:

  * `nsepython` — modern, actively maintained NSE scraper
  * `nsetools`  — older, simpler NSE client (used as NSE fallback)
  * `bsedata`   — BSE scraper

Resolution order for `get_stock_info(symbol)`:
  1. If `symbol` is all digits, treat it as a BSE security code and go
     straight to BSE (e.g. "500325" → Reliance).
  2. Otherwise try NSE via `nsepython.nse_eq`.
  3. If nsepython fails, try `nsetools.Nse().get_quote()`.
  4. If both NSE paths fail and `bse_fallback` is enabled, resolve the
     symbol to a BSE scrip code by name and fetch from BSE.

All three libraries are imported lazily so the adapter degrades
gracefully if any of them is uninstalled or fails to import.

Each successful lookup returns a *normalized* dict with the same keys
regardless of which library produced it — see `_NORMALIZED_KEYS` below.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
from typing import Any

from adapters.base_adapter import BaseStockAdapter

logger = logging.getLogger(__name__)

# --- Optional imports (each library is independently optional) --------------

try:
    from nsepython import nse_eq  # type: ignore

    _HAS_NSEPYTHON = True
except Exception as _e:  # noqa: BLE001
    nse_eq = None  # type: ignore[assignment]
    _HAS_NSEPYTHON = False
    logger.debug("nsepython unavailable: %s", _e)

try:
    from nsetools import Nse  # type: ignore

    _HAS_NSETOOLS = True
except Exception as _e:  # noqa: BLE001
    Nse = None  # type: ignore[assignment]
    _HAS_NSETOOLS = False
    logger.debug("nsetools unavailable: %s", _e)

try:
    from bsedata.bse import BSE  # type: ignore

    _HAS_BSEDATA = True
except Exception as _e:  # noqa: BLE001
    BSE = None  # type: ignore[assignment]
    _HAS_BSEDATA = False
    logger.debug("bsedata unavailable: %s", _e)


_NORMALIZED_KEYS = (
    "source",
    "exchange",
    "symbol",
    "companyName",
    "industry",
    "isin",
    "lastPrice",
    "change",
    "pChange",
    "open",
    "previousClose",
    "dayHigh",
    "dayLow",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    "marketCap",
    "lastUpdateTime",
    "currency",
    # --- NSE-specific enrichment (None for BSE/nsetools) -----------------
    # All of these come from the same nse_eq response we already make, so
    # they cost nothing extra. See `_fetch_via_nsepython` for the mapping.
    "vwap",
    "upperCircuit",
    "lowerCircuit",
    "priceBand",
    "tickSize",
    "basePrice",
    "listingDate",
    "faceValue",
    "issuedSize",
    "surveillance",      # ASM / GSM stage if applicable
    "tradingStatus",
    "isFnoSec",
    "isEtfSec",
    "sectorPE",
    "symbolPE",
    "nseSector",
    "nseMacro",
    "nseBasicIndustry",
)


class NseBseAdapter(BaseStockAdapter):
    """Fetches live Indian equity quotes by trying NSE, then BSE."""

    def __init__(self, bse_fallback: bool = True) -> None:
        self._bse_fallback = bse_fallback
        # IMPORTANT: both `nsetools.Nse()` and `bsedata.BSE()` make live
        # HTTP calls during construction (NSE warms a cookie session,
        # BSE optionally pulls the scrip-code list). If those calls fail
        # at import time we'd take down the entire agent. So we defer
        # construction to the first actual lookup and wrap it in
        # try/except so failures degrade gracefully.
        self._nse: Any = None
        self._bse: Any = None
        self._nse_init_attempted = False
        self._bse_init_attempted = False
        # BSE name→code map is populated lazily on first fallback lookup.
        self._bse_codes_loaded = False

    def _get_nse(self) -> Any:
        if self._nse is not None or self._nse_init_attempted:
            return self._nse
        self._nse_init_attempted = True
        if not _HAS_NSETOOLS or Nse is None:
            return None
        try:
            self._nse = Nse()
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to initialize nsetools.Nse: %s", e)
            self._nse = None
        return self._nse

    def _get_bse(self) -> Any:
        if self._bse is not None or self._bse_init_attempted:
            return self._bse
        self._bse_init_attempted = True
        if not _HAS_BSEDATA or BSE is None:
            return None
        try:
            self._bse = BSE()
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to initialize bsedata.BSE: %s", e)
            self._bse = None
        return self._bse

    # --- Public API ---------------------------------------------------------

    def get_stock_info(self, stock_name: str) -> dict:
        symbol = (stock_name or "").strip().upper()
        if not symbol:
            return {"error": "Empty symbol passed to NseBseAdapter."}

        # All-digit input → BSE security code (e.g. "500325").
        if symbol.isdigit():
            return self._fetch_bse_by_code(symbol)

        errors: list[str] = []

        if _HAS_NSEPYTHON:
            result = self._fetch_via_nsepython(symbol)
            if "error" not in result:
                return result
            errors.append(f"nsepython: {result['error']}")
        else:
            errors.append("nsepython: not installed")

        if _HAS_NSETOOLS and self._get_nse() is not None:
            result = self._fetch_via_nsetools(symbol)
            if "error" not in result:
                return result
            errors.append(f"nsetools: {result['error']}")
        else:
            errors.append("nsetools: unavailable")

        if self._bse_fallback and self._get_bse() is not None:
            result = self._fetch_bse_by_name(symbol)
            if "error" not in result:
                return result
            errors.append(f"bsedata: {result['error']}")
        elif self._get_bse() is None:
            errors.append("bsedata: unavailable")

        return {
            "error": f"All NSE/BSE lookups failed for '{symbol}'.",
            "details": errors,
        }

    # --- NSE: nsepython -----------------------------------------------------

    def _fetch_via_nsepython(self, symbol: str) -> dict:
        # nsepython unconditionally `print()`s notices like
        # "Please use nse_fno() function to reduce latency." straight to
        # stdout, which leaks into the agent's chat output. Redirect
        # stdout while the call runs so those notices are swallowed.
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                raw = nse_eq(symbol)  # type: ignore[misc]
        except Exception as e:  # noqa: BLE001
            return {"error": f"nse_eq raised: {e}"}

        if not isinstance(raw, dict) or "priceInfo" not in raw:
            return {"error": "Unexpected response shape from nsepython."}

        info = raw.get("info") or {}
        price = raw.get("priceInfo") or {}
        metadata = raw.get("metadata") or {}
        security = raw.get("securityInfo") or {}
        industry_info = raw.get("industryInfo") or {}
        wh = price.get("weekHighLow") or {}
        idh = price.get("intraDayHighLow") or {}

        # `surveillance` is itself a small dict in modern nsepython —
        # collapse to its `surv` (stage) field where present, else None.
        surveillance_raw = security.get("surveillance")
        if isinstance(surveillance_raw, dict):
            surveillance = surveillance_raw.get("surv") or surveillance_raw.get(
                "desc"
            )
        else:
            surveillance = surveillance_raw

        return self._normalize(
            source="NSE (nsepython)",
            exchange="NSE",
            symbol=info.get("symbol") or symbol,
            companyName=info.get("companyName"),
            industry=info.get("industry"),
            isin=info.get("isin"),
            lastPrice=price.get("lastPrice"),
            change=price.get("change"),
            pChange=price.get("pChange"),
            open=price.get("open"),
            previousClose=price.get("previousClose"),
            dayHigh=idh.get("max"),
            dayLow=idh.get("min"),
            fiftyTwoWeekHigh=wh.get("max"),
            fiftyTwoWeekLow=wh.get("min"),
            marketCap=None,  # nsepython doesn't return market cap here
            lastUpdateTime=metadata.get("lastUpdateTime"),
            # --- NSE enrichment from the SAME response --------------------
            vwap=price.get("vwap"),
            upperCircuit=price.get("upperCP"),
            lowerCircuit=price.get("lowerCP"),
            priceBand=price.get("pPriceBand"),
            tickSize=price.get("tickSize"),
            basePrice=price.get("basePrice"),
            listingDate=info.get("listingDate") or metadata.get("listingDate"),
            faceValue=security.get("faceValue"),
            issuedSize=security.get("issuedSize"),
            surveillance=surveillance,
            tradingStatus=security.get("tradingStatus"),
            isFnoSec=info.get("isFNOSec"),
            isEtfSec=info.get("isETFSec"),
            sectorPE=metadata.get("pdSectorPe"),
            symbolPE=metadata.get("pdSymbolPe"),
            nseSector=industry_info.get("sector"),
            nseMacro=industry_info.get("macro"),
            nseBasicIndustry=industry_info.get("basicIndustry"),
        )

    # --- NSE: nsetools ------------------------------------------------------

    def _fetch_via_nsetools(self, symbol: str) -> dict:
        nse = self._get_nse()
        if nse is None:
            return {"error": "nsetools unavailable."}
        try:
            quote = nse.get_quote(symbol)
        except Exception as e:  # noqa: BLE001
            return {"error": f"nsetools raised: {e}"}

        if not quote:
            return {"error": f"nsetools returned no data for '{symbol}'."}

        return self._normalize(
            source="NSE (nsetools)",
            exchange="NSE",
            symbol=quote.get("symbol") or symbol,
            companyName=quote.get("companyName"),
            industry=quote.get("industry"),
            isin=quote.get("isinCode"),
            lastPrice=quote.get("lastPrice"),
            change=quote.get("change"),
            pChange=quote.get("pChange"),
            open=quote.get("open"),
            previousClose=quote.get("previousClose"),
            dayHigh=quote.get("dayHigh"),
            dayLow=quote.get("dayLow"),
            fiftyTwoWeekHigh=quote.get("high52"),
            fiftyTwoWeekLow=quote.get("low52"),
            marketCap=None,
            lastUpdateTime=quote.get("secDate"),
        )

    # --- BSE: bsedata -------------------------------------------------------

    def _ensure_bse_codes(self) -> bool:
        """Lazily populate the BSE name→code map. Returns True on success."""
        bse = self._get_bse()
        if bse is None:
            return False
        if self._bse_codes_loaded:
            return True
        try:
            bse.updateScripCodes()
            self._bse_codes_loaded = True
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to update BSE scrip codes: %s", e)
            return False

    def _fetch_bse_by_code(self, code: str) -> dict:
        bse = self._get_bse()
        if bse is None:
            return {"error": "bsedata unavailable."}
        try:
            quote = bse.getQuote(code)
        except Exception as e:  # noqa: BLE001
            return {"error": f"bsedata raised: {e}"}
        if not quote:
            return {"error": f"bsedata returned no data for code '{code}'."}
        return self._normalize_bse(quote, code)

    def _fetch_bse_by_name(self, symbol: str) -> dict:
        bse = self._get_bse()
        if bse is None:
            return {"error": "bsedata unavailable."}
        if not self._ensure_bse_codes():
            return {"error": "Could not load BSE scrip code table."}

        try:
            codes = bse.getScripCodes() or {}
        except Exception as e:  # noqa: BLE001
            return {"error": f"bsedata code lookup raised: {e}"}

        target = symbol.upper()
        # codes is {scrip_code: company_name} — find an exact-ish match.
        match_code: str | None = None
        for code, name in codes.items():
            if not isinstance(name, str):
                continue
            up = name.upper()
            if up == target or target in up.split():
                match_code = str(code)
                break
        if match_code is None:
            # Loose contains-match as a last resort.
            for code, name in codes.items():
                if isinstance(name, str) and target in name.upper():
                    match_code = str(code)
                    break
        if match_code is None:
            return {"error": f"No BSE scrip code found for '{symbol}'."}

        return self._fetch_bse_by_code(match_code)

    def _normalize_bse(self, quote: dict, code: str) -> dict:
        market_cap_full = quote.get("marketCapFull")
        return self._normalize(
            source="BSE (bsedata)",
            exchange="BSE",
            symbol=quote.get("securityID") or code,
            companyName=quote.get("companyName"),
            industry=quote.get("industry"),
            isin=None,
            lastPrice=quote.get("currentValue"),
            change=quote.get("change"),
            pChange=quote.get("pChange"),
            open=quote.get("previousOpen"),
            previousClose=quote.get("previousClose"),
            dayHigh=quote.get("dayHigh"),
            dayLow=quote.get("dayLow"),
            fiftyTwoWeekHigh=quote.get("52weekHigh"),
            fiftyTwoWeekLow=quote.get("52weekLow"),
            marketCap=market_cap_full,
            lastUpdateTime=quote.get("updatedOn"),
        )

    # --- Helpers ------------------------------------------------------------

    @staticmethod
    def _normalize(**kwargs: Any) -> dict:
        """Return a dict containing only the canonical normalized keys."""
        out = {k: kwargs.get(k) for k in _NORMALIZED_KEYS}
        out["currency"] = "INR"
        return out
