"""Tavily-backed adapter for live news / qualitative stock context.

`YFinanceAdapter` is great for structured fundamentals (price, P/E, market
cap, ...), but it does not surface recent news, market sentiment, or
qualitative analyst commentary. Tavily is a web-search API tuned for LLM
agents and fills that gap.

The adapter follows the same `BaseStockAdapter` contract as the yfinance
adapter so the tool layer can treat both data sources symmetrically.

Inspired by:
    https://github.com/shubhamveida/langraph-example/blob/main/main.py
"""

from __future__ import annotations

import os
from typing import Any

from tavily import TavilyClient

from adapters.base_adapter import BaseStockAdapter


class TavilyAdapter(BaseStockAdapter):
    """Fetches recent news / qualitative info about a ticker via Tavily."""

    def __init__(self, max_results: int = 3, search_depth: str = "basic") -> None:
        self._max_results = max_results
        self._search_depth = search_depth

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            # Defer the error until first call so importing the module
            # doesn't blow up environments that only use yfinance.
            self._client: TavilyClient | None = None
            self._init_error: str | None = (
                "TAVILY_API_KEY is not set. Add it to your .env file."
            )
        else:
            self._client = TavilyClient(api_key=api_key)
            self._init_error = None

    def get_stock_info(self, stock_name: str) -> dict:
        """Search the web for recent news / analysis about `stock_name`.

        Returns a dict with the original query, Tavily's optional
        synthesized `answer`, and a list of `results` (title, url, snippet).
        On failure, returns a dict with an `error` key.
        """
        if self._client is None:
            return {"error": self._init_error or "Tavily client not initialized."}

        query = (
            f"latest financial news, earnings, and analyst sentiment for {stock_name}"
        )

        try:
            response: Any = self._client.search(
                query=query,
                max_results=self._max_results,
                search_depth=self._search_depth,
                include_answer=True,
            )
        except Exception as e:  # noqa: BLE001 — surface any client/network failure
            return {"error": f"Failed to fetch data from Tavily: {e}"}

        if not isinstance(response, dict):
            return {"error": f"Unexpected Tavily response type: {type(response).__name__}"}

        raw_results = response.get("results") or []
        results = [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "content": r.get("content"),
                "score": r.get("score"),
                "published_date": r.get("published_date"),
            }
            for r in raw_results
            if isinstance(r, dict)
        ]

        return {
            "query": query,
            "answer": response.get("answer"),
            "results": results,
        }
