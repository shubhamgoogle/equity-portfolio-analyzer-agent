"""ADK-compatible tools that wrap the `adapters/` data-source layer.

Each function here is exposed to the LLM as a tool. The function name
becomes the tool name and the docstring becomes its description, so keep
them clear and user-facing.

To add a new data source, implement a new adapter under `adapters/` that
satisfies `BaseStockAdapter`, then add a thin wrapper function below.
"""

from __future__ import annotations

import json
from typing import Any

from adapters.tavily_adapter import TavilyAdapter
from adapters.yfinance_adapter import YFinanceAdapter

# A single adapter instance is fine — yfinance is stateless per-call.
_yfinance_adapter = YFinanceAdapter()
# Tavily client is constructed once and reused across tool calls.
_tavily_adapter = TavilyAdapter()


def _to_jsonable(value: Any) -> Any:
    """Coerce arbitrary values (e.g. numpy scalars) into JSON-safe types.

    yfinance occasionally returns numpy types; ADK will serialize tool
    results to JSON when sending them back to the model, so we normalize
    here to avoid surprises.
    """
    return json.loads(json.dumps(value, default=str))


def get_stock_info(symbol: str) -> dict:
    """Fetch fundamentals for a stock, ETF, or mutual fund by ticker symbol.

    Use this whenever the user asks about a specific instrument. It returns
    Yahoo Finance fundamentals such as price, market cap, sector, industry,
    trailing/forward P/E, dividend yield, 52-week range, and more.

    Args:
        symbol: The Yahoo Finance ticker symbol. Examples:
            - US stock: "AAPL", "GOOGL", "MSFT"
            - US mutual fund: "VFIAX", "FXAIX"
            - Indian mutual fund: "0P0000XW8M.BO"
            - ETF: "SPY", "QQQ"

    Returns:
        A dict of fundamentals on success, or a dict with an `error` key
        describing the failure (e.g. unknown symbol, network error).
    """
    info = _yfinance_adapter.get_stock_info(symbol)
    return _to_jsonable(info)


def get_stock_news(query: str) -> dict:
    """Fetch recent news, headlines, and qualitative analysis about a stock.

    Use this *in addition to* `get_stock_info` whenever the user wants
    context that fundamentals alone can't answer: recent earnings,
    analyst upgrades/downgrades, regulatory events, management changes,
    macro headlines, or general market sentiment.

    Args:
        query: A ticker symbol (e.g. "AAPL") or a free-form company /
            fund name (e.g. "Apple Inc", "Parag Parikh Flexi Cap").
            A plain ticker works well — the adapter expands it into a
            news-oriented search query internally.

    Returns:
        A dict with:
            - `query`: the search query that was actually run
            - `answer`: an optional one-paragraph synthesized summary
              from Tavily (may be None)
            - `results`: a list of {title, url, content, score,
              published_date} snippets from recent web sources

        On failure, returns a dict with an `error` key (e.g. missing
        API key, network error).
    """
    info = _tavily_adapter.get_stock_info(query)
    return _to_jsonable(info)


ALL_TOOLS = [get_stock_info, get_stock_news]
