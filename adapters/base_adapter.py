from abc import ABC, abstractmethod

class BaseStockAdapter(ABC):
    @abstractmethod
    def get_stock_info(self, stock_name: str) -> dict:
        """
        Fetches stock information for a given stock name.
        Returns a dictionary with the stock information.
        """
        pass
