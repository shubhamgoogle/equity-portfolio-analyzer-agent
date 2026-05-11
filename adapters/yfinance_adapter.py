import yfinance as yf
from adapters.base_adapter import BaseStockAdapter

class YFinanceAdapter(BaseStockAdapter):
    def get_stock_info(self, stock_name: str) -> dict:
        """
        Fetches stock information using yfinance.
        """
        try:
            ticker = yf.Ticker(stock_name)
            # yfinance returns a dict-like object for info
            info = ticker.info
            return info
        except Exception as e:
            return {"error": f"Failed to fetch data from yfinance: {str(e)}"}
