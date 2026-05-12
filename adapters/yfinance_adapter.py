"""Yahoo Finance adapter — global fundamentals + business summary.

Returns three logical groups in a single dict:

1. The full Yahoo `ticker.info` payload (~150-180 keys) for backwards
   compatibility — the agent already knows how to read these.
2. An `extras.statements` block: per-fiscal-year history (up to 5y) of
   the line items needed to build a trajectory view (revenue, profit,
   EBITDA, EPS, operating/investing/financing CF, capex, debt, equity,
   working capital, ...).
3. An `extras.derived` block: ratios and CAGRs that aren't in `.info`
   but are trivially computable from (2) — ROCE, interest coverage,
   3y/5y revenue & profit & EPS & dividend CAGRs.
4. An `extras.dividend_history` / `extras.split_history` / `extras.analyst_trend`
   block: clean lists with the most recent N events plus a trend summary.

All of (2)-(4) come from FREE yfinance endpoints (`.financials`,
`.balance_sheet`, `.cashflow`, `.dividends`, `.splits`, `.recommendations`).
Each one is wrapped in try/except so a partial Yahoo outage degrades
gracefully — the existing `info` dict is still returned.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd
import yfinance as yf

from adapters.base_adapter import BaseStockAdapter

# yfinance logs every HTTP 404 / rate-limit / quote-not-found at ERROR
# level *before* raising, which pollutes the terminal even though our
# adapter catches the exception and wraps it as {"error": ...}. We
# already surface failures in the return value, so demote yfinance's
# loggers to CRITICAL so nothing below that level leaks to stderr.
for _logger_name in ("yfinance", "yfinance.ticker", "yfinance.utils", "yfinance.data"):
    logging.getLogger(_logger_name).setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

# Yahoo's statement DataFrames use long-form English row labels. These
# are the rows we care about, keyed by the canonical name we expose.
_INCOME_ROWS = {
    "revenue": ("Total Revenue", "Operating Revenue"),
    "gross_profit": ("Gross Profit",),
    "operating_income": ("Operating Income",),
    "ebitda": ("EBITDA", "Normalized EBITDA"),
    "ebit": ("EBIT",),
    "net_income": ("Net Income", "Net Income Common Stockholders"),
    "interest_expense": ("Interest Expense", "Interest Expense Non Operating"),
    "tax_provision": ("Tax Provision",),
    "basic_eps": ("Basic EPS",),
    "diluted_eps": ("Diluted EPS",),
}

_BALANCE_ROWS = {
    "total_assets": ("Total Assets",),
    "total_liabilities": ("Total Liabilities Net Minority Interest",),
    "stockholders_equity": ("Stockholders Equity", "Common Stock Equity"),
    "total_debt": ("Total Debt",),
    "net_debt": ("Net Debt",),
    "long_term_debt": ("Long Term Debt",),
    "current_debt": ("Current Debt",),
    "cash": ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"),
    "current_assets": ("Current Assets",),
    "current_liabilities": ("Current Liabilities",),
    "working_capital": ("Working Capital",),
    "invested_capital": ("Invested Capital",),
    "tangible_book_value": ("Tangible Book Value",),
}

_CASHFLOW_ROWS = {
    "operating_cash_flow": ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities"),
    "investing_cash_flow": ("Investing Cash Flow", "Cash Flow From Continuing Investing Activities"),
    "financing_cash_flow": ("Financing Cash Flow", "Cash Flow From Continuing Financing Activities"),
    "free_cash_flow": ("Free Cash Flow",),
    "capital_expenditure": ("Capital Expenditure",),
    "cash_dividends_paid": ("Cash Dividends Paid",),
}


def _is_clean_number(x: Any) -> bool:
    """yfinance frequently leaves NaN cells where data isn't reported."""
    if x is None:
        return False
    try:
        f = float(x)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(f) or math.isinf(f))


def _extract_series(df: pd.DataFrame | None, label_candidates: tuple[str, ...]) -> list[Any]:
    """Look up a row by any of the candidate labels and return its values
    in the same column order as `df.columns`. Returns empty list if df
    is missing, empty, or no candidate matched."""
    if df is None or df.empty:
        return []
    for label in label_candidates:
        if label in df.index:
            row = df.loc[label]
            return [float(v) if _is_clean_number(v) else None for v in row.values]
    return []


def _cagr(values: list[Any], years: int) -> float | None:
    """Compute CAGR from an *ordered-newest-first* list (the convention
    Yahoo's statement DataFrames use). Skips leading `None` cells so a
    yet-to-be-reported fiscal year doesn't kill the calculation: we use
    the most recent valid value as the endpoint and look back `years`
    periods from THAT index. Returns None if we don't have enough valid
    datapoints, the start point is non-positive, or the direction
    crosses zero (e.g. losses → profits)."""
    if not values or years <= 0:
        return None
    end_idx = next(
        (i for i, v in enumerate(values) if _is_clean_number(v)), None
    )
    if end_idx is None:
        return None
    start_idx = end_idx + years
    if start_idx >= len(values):
        return None
    end = values[end_idx]
    start = values[start_idx]
    if not _is_clean_number(start):
        return None
    if start <= 0 or end <= 0:
        return None
    return (end / start) ** (1.0 / years) - 1.0


def _format_period(col: Any) -> str | None:
    """Format a DataFrame column label (typically a Timestamp) as YYYY-MM-DD."""
    if hasattr(col, "strftime"):
        try:
            return col.strftime("%Y-%m-%d")
        except Exception:  # noqa: BLE001
            return None
    if col is None:
        return None
    return str(col)


class YFinanceAdapter(BaseStockAdapter):
    def get_stock_info(self, stock_name: str) -> dict:
        """Fetches stock information + multi-year statements from yfinance.

        Returns the raw `ticker.info` dict augmented with an `extras` key
        that summarizes financial statements, dividend / split history,
        analyst recommendation trend, and a handful of derived ratios
        (CAGRs, ROCE, interest coverage). All extras are produced from
        FREE yfinance endpoints.
        """
        try:
            ticker = yf.Ticker(stock_name)
        except Exception as e:  # noqa: BLE001
            return {"error": f"Failed to construct yfinance.Ticker: {e}"}

        try:
            info: dict[str, Any] = dict(ticker.info or {})
        except Exception as e:  # noqa: BLE001
            return {"error": f"Failed to fetch data from yfinance: {e}"}

        extras: dict[str, Any] = {}
        extra_errors: list[str] = []

        statements = self._fetch_statements(ticker, extra_errors)
        if statements:
            extras["statements"] = statements
            derived = self._derive_metrics(statements, info)
            if derived:
                extras["derived"] = derived

        div_history = self._fetch_dividends(ticker, extra_errors)
        if div_history:
            extras["dividend_history"] = div_history

        split_history = self._fetch_splits(ticker, extra_errors)
        if split_history:
            extras["split_history"] = split_history

        analyst = self._fetch_recommendations(ticker, extra_errors)
        if analyst:
            extras["analyst_trend"] = analyst

        if extras:
            info["extras"] = extras
        if extra_errors:
            # Surface partial failures without overriding info["error"]
            # (Yahoo can return a perfectly good `info` even when the
            # statement endpoints rate-limit us, so don't poison the
            # primary payload — just attach diagnostic info).
            info["_extras_errors"] = extra_errors
        return info

    # --- Statements ---------------------------------------------------------

    def _fetch_statements(self, ticker: yf.Ticker, errors: list[str]) -> dict | None:
        """Return up to 5y of key line items from income/balance/cashflow."""
        try:
            fin = ticker.financials
        except Exception as e:  # noqa: BLE001
            errors.append(f"financials: {e}")
            fin = None
        try:
            bs = ticker.balance_sheet
        except Exception as e:  # noqa: BLE001
            errors.append(f"balance_sheet: {e}")
            bs = None
        try:
            cf = ticker.cashflow
        except Exception as e:  # noqa: BLE001
            errors.append(f"cashflow: {e}")
            cf = None

        # If literally everything failed, skip the whole block.
        if all(df is None or df.empty for df in (fin, bs, cf)):
            return None

        # Use whichever statement has columns to source the period labels.
        # Yahoo orders newest-first, so years[0] is the most recent FY.
        period_source = next(
            (df for df in (fin, bs, cf) if df is not None and not df.empty), None
        )
        periods = (
            [_format_period(c) for c in period_source.columns]
            if period_source is not None
            else []
        )

        block: dict[str, Any] = {"periods": periods}

        income: dict[str, list[Any]] = {}
        for k, candidates in _INCOME_ROWS.items():
            vals = _extract_series(fin, candidates)
            if vals:
                income[k] = vals
        if income:
            block["income"] = income

        balance: dict[str, list[Any]] = {}
        for k, candidates in _BALANCE_ROWS.items():
            vals = _extract_series(bs, candidates)
            if vals:
                balance[k] = vals
        if balance:
            block["balance"] = balance

        cashflow: dict[str, list[Any]] = {}
        for k, candidates in _CASHFLOW_ROWS.items():
            vals = _extract_series(cf, candidates)
            if vals:
                cashflow[k] = vals
        if cashflow:
            block["cashflow"] = cashflow

        # If we ended up with nothing useful, drop the empty wrapper.
        if not any(k in block for k in ("income", "balance", "cashflow")):
            return None
        return block

    # --- Derived metrics ----------------------------------------------------

    @staticmethod
    def _derive_metrics(statements: dict, info: dict) -> dict | None:
        """Compute ratios + CAGRs from the statements block."""
        income = statements.get("income") or {}
        balance = statements.get("balance") or {}
        cashflow = statements.get("cashflow") or {}

        out: dict[str, Any] = {}

        def latest(series: list[Any]) -> float | None:
            for v in series:
                if _is_clean_number(v):
                    return float(v)
            return None

        ebit_latest = latest(income.get("ebit", []))
        int_exp_latest = latest(income.get("interest_expense", []))
        if ebit_latest is not None and int_exp_latest:
            # interest_expense is typically reported as a positive cost,
            # but Yahoo occasionally flips the sign. Take the magnitude.
            denom = abs(int_exp_latest)
            if denom > 0:
                out["interest_coverage_latest"] = round(ebit_latest / denom, 2)

        equity_latest = latest(balance.get("stockholders_equity", []))
        debt_latest = latest(balance.get("total_debt", []))
        if (
            ebit_latest is not None
            and equity_latest is not None
            and debt_latest is not None
            and (equity_latest + debt_latest) > 0
        ):
            out["roce_latest"] = round(
                ebit_latest / (equity_latest + debt_latest), 4
            )

        wc_latest = latest(balance.get("working_capital", []))
        if wc_latest is not None:
            out["working_capital_latest"] = wc_latest

        capex_latest = latest(cashflow.get("capital_expenditure", []))
        if capex_latest is not None:
            out["capex_latest"] = capex_latest
        inv_cf_latest = latest(cashflow.get("investing_cash_flow", []))
        if inv_cf_latest is not None:
            out["investing_cash_flow_latest"] = inv_cf_latest
        fin_cf_latest = latest(cashflow.get("financing_cash_flow", []))
        if fin_cf_latest is not None:
            out["financing_cash_flow_latest"] = fin_cf_latest

        net_income_latest = latest(income.get("net_income", []))
        fcf_latest = latest(cashflow.get("free_cash_flow", []))
        if (
            net_income_latest is not None
            and fcf_latest is not None
            and net_income_latest > 0
        ):
            # FCF / NI: how much accounting profit converts to actually
            # spendable cash after reinvestment. >0.8 is healthy, <0.3
            # is a yellow flag (high accruals or heavy capex burden).
            out["cash_conversion_latest"] = round(fcf_latest / net_income_latest, 3)

        # Yahoo's free statements top out at 5 fiscal years, so 5y CAGR
        # (which needs values at year 0 and year 5 = 6 datapoints) isn't
        # computable. 3y and 4y are.
        for label, key in (
            ("revenue", "revenue"),
            ("net_income", "profit"),
            ("diluted_eps", "eps"),
        ):
            series = income.get(label, [])
            for y in (3, 4):
                cagr = _cagr(series, y)
                if cagr is not None:
                    out[f"{key}_cagr_{y}y"] = round(cagr, 4)

        return out or None

    # --- Dividend history --------------------------------------------------

    @staticmethod
    def _fetch_dividends(ticker: yf.Ticker, errors: list[str]) -> dict | None:
        try:
            divs = ticker.dividends
        except Exception as e:  # noqa: BLE001
            errors.append(f"dividends: {e}")
            return None
        if divs is None or divs.empty:
            return None

        # Recent 8 payouts is plenty for the LLM to spot a trend.
        recent = list(divs.tail(8).items())
        events = [
            {
                "date": ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts),
                "amount": float(amt),
            }
            for ts, amt in recent
            if _is_clean_number(amt)
        ]

        # Annual dividend totals → dividend CAGR.
        by_year: dict[int, float] = {}
        for ts, amt in divs.items():
            if not _is_clean_number(amt):
                continue
            year = ts.year if hasattr(ts, "year") else None
            if year is None:
                continue
            by_year[year] = by_year.get(year, 0.0) + float(amt)
        years_sorted = sorted(by_year.keys(), reverse=True)
        # Skip the current calendar year if it's clearly partial.
        annual_series = [by_year[y] for y in years_sorted]

        out: dict[str, Any] = {"recent": events}
        for n in (3, 5):
            cagr = _cagr(annual_series, n)
            if cagr is not None:
                out[f"dividend_cagr_{n}y"] = round(cagr, 4)
        return out

    # --- Split history -----------------------------------------------------

    @staticmethod
    def _fetch_splits(ticker: yf.Ticker, errors: list[str]) -> list | None:
        try:
            sp = ticker.splits
        except Exception as e:  # noqa: BLE001
            errors.append(f"splits: {e}")
            return None
        if sp is None or sp.empty:
            return None
        events = [
            {
                "date": ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts),
                "ratio": float(r),
            }
            for ts, r in sp.items()
            if _is_clean_number(r)
        ]
        return events or None

    # --- Analyst recommendation trend --------------------------------------

    @staticmethod
    def _fetch_recommendations(ticker: yf.Ticker, errors: list[str]) -> dict | None:
        try:
            rec = ticker.recommendations
        except Exception as e:  # noqa: BLE001
            errors.append(f"recommendations: {e}")
            return None
        if rec is None or rec.empty:
            return None

        # `recommendations` on modern yfinance has `period` (e.g. "0m",
        # "-1m", "-2m", "-3m") and counts for strongBuy/buy/hold/sell/strongSell.
        # Surface the four most recent rows as-is so the LLM can spot
        # rating drift quarter-over-quarter.
        cols = [
            c for c in ("period", "strongBuy", "buy", "hold", "sell", "strongSell")
            if c in rec.columns
        ]
        rows: list[dict] = []
        for _, row in rec.head(4).iterrows():
            cleaned = {}
            for c in cols:
                v = row[c]
                if isinstance(v, (int, float)) and _is_clean_number(v):
                    cleaned[c] = int(v) if c != "period" else v
                else:
                    cleaned[c] = v
            rows.append(cleaned)
        return {"by_period": rows} if rows else None
