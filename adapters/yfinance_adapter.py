"""Yahoo Finance adapter — global fundamentals + business summary."""

from __future__ import annotations

import logging

import yfinance as yf

from adapters.base_adapter import BaseStockAdapter

# yfinance logs every HTTP 404 / rate-limit / quote-not-found at ERROR
# level *before* raising, which pollutes the terminal even though our
# adapter catches the exception and wraps it as {"error": ...}. We
# already surface failures in the return value, so demote yfinance's
# loggers to CRITICAL so nothing below that level leaks to stderr.
for _logger_name in ("yfinance", "yfinance.ticker", "yfinance.utils", "yfinance.data"):
    logging.getLogger(_logger_name).setLevel(logging.CRITICAL)


class YFinanceAdapter(BaseStockAdapter):
    def get_stock_info(self, stock_name: str) -> dict:
        """Fetches stock information using yfinance."""
        try:
            ticker = yf.Ticker(stock_name)
            info = ticker.info
            return info
        except Exception as e:
            return {"error": f"Failed to fetch data from yfinance: {str(e)}"}
