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

import logging
import os
import time
from typing import Any

from tavily import TavilyClient

from adapters.base_adapter import BaseStockAdapter

logger = logging.getLogger(__name__)


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

        logger.info("Tavily search start: symbol=%r", stock_name)
        started = time.perf_counter()
        try:
            response: Any = self._client.search(
                query=query,
                max_results=self._max_results,
                search_depth=self._search_depth,
                include_answer=True,
            )
        except Exception as e:  # noqa: BLE001 — surface any client/network failure
            logger.warning(
                "Tavily search failed for %r after %.0fms: %s",
                stock_name,
                (time.perf_counter() - started) * 1000,
                e,
            )
            return {"error": f"Failed to fetch data from Tavily: {e}"}

        elapsed_ms = (time.perf_counter() - started) * 1000

        if not isinstance(response, dict):
            logger.warning(
                "Tavily returned unexpected shape %s for %r after %.0fms",
                type(response).__name__,
                stock_name,
                elapsed_ms,
            )
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

        # Single visible breadcrumb per successful Tavily call. The SDK
        # itself is silent on success, so without this line a working
        # Tavily call leaves no trace in the agent's stdout.
        logger.info(
            "Tavily search ok: symbol=%r results=%d answer=%s elapsed=%.0fms",
            stock_name,
            len(results),
            "yes" if response.get("answer") else "no",
            elapsed_ms,
        )

        return {
            "query": query,
            "answer": response.get("answer"),
            "results": results,
        }
