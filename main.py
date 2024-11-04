from utils import download_history
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class StockParams:
    def __init__(self) -> None:
        pass
         

class Stock:
    def __init__(self, ticker, start=None, end=None, period='1d', interval='1m'):
        self.__ticker = ticker
        self.__data = download_history(self.__ticker,
                                     start=start, end=end,
                                     period=period, interval=interval)

    def plot(self, key):
        plt.plot(self[key])
        plt.show()

    def __getitem__(self, key):
        return self.__data[key] 


class Indicator(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, stock: Stock):
        raise NotImplementedError


if __name__ == "__main__":
    AAPL = Stock('AAPL')
    AAPL.plot('Close')
