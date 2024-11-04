import pandas as pd 
import pandas_ta as ta
import matplotlib.pyplot as plt
from utils import download_history
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
        data = self.__data[key]
        if data is None:
            return

        plt.plot(data)
        plt.show()

    def __getitem__(self, key):
        try:
            return self.__data[key] 
        except KeyError:
            return None

class Indicator(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, stock: Stock) -> pd.DataFrame:
        raise NotImplementedError


class RSI(Indicator):
    def __init__(self, period: int = 14) -> None:
        super().__init__()
        self.__period = period
        print("hello world")

    def __call__(self, stock: Stock) -> pd.DataFrame:
        assert stock['Close'] is not None, 'Close column is missing' 
        close = stock['Close']
        dataframe = ta.rsi(close, offset=self.__period)

        if dataframe is None:
            return pd.DataFrame()

        return dataframe


if __name__ == "__main__":
    AAPL = Stock('AAPL')

    rsi = RSI()
    rsi_df = rsi(AAPL)

    plt.plot(AAPL['Close'])
    plt.plot(rsi_df)
    plt.show()



