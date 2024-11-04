import pandas as pd 
import pandas_ta as ta
import matplotlib.pyplot as plt
from pandas_ta import volume
from utils import download_history
from abc import ABC, abstractmethod
from typing import List


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
    name: str
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, stock: Stock) -> pd.DataFrame:
        raise NotImplementedError


class RSI(Indicator):
    name = 'RSI'
    def __init__(self, period: int = 14) -> None:
        super().__init__()
        self.__period = period
        print("hello world")

    def __call__(self, stock: Stock) -> pd.DataFrame:
        assert stock['Close'] is not None, 'Close column is missing' 
        close = stock['Close']
        if close is None:
            return pd.DataFrame()
        dataframe = ta.rsi(close, offset=self.__period)

        if dataframe is None:
            return pd.DataFrame()

        return dataframe

class MACD(Indicator):
    name = 'MACD'

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, stock: Stock) -> pd.DataFrame:
        """
        Produces 3 time series: MACD Line, Signal Line, and MACD Histogram.
        """
        assert stock['Close'] is not None, 'Close column is missing' 

        close = stock['Close']
        if close is None:
            return pd.DataFrame()
        dataframe = ta.macd(close)

        if dataframe is None:
            return pd.DataFrame()

        return dataframe


class StochasticOscillator(Indicator):
    name = "Stochastic Oscillator"
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, stock: Stock) -> pd.DataFrame:
        assert stock['Close'] is not None, 'Close column is missing' 
        assert stock['High'] is not None, 'Close column is missing' 
        assert stock['Low'] is not None, 'Close column is missing' 

        close = stock['Close']
        high = stock['High']
        low = stock['Low']
        if close is None:
            return pd.DataFrame()

        dataframe = ta.stoch(high, low, close) # type: ignore

        if dataframe is None:
            return pd.DataFrame()

        return dataframe

class OBV(Indicator):
    name = "OBV"
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, stock: Stock) -> pd.DataFrame:
        assert stock['Close'] is not None, 'Close column is missing' 
        assert stock['Volume'] is not None, 'Volume column is missing' 

        close = stock['Close']
        if close is None:
            return pd.DataFrame()

        volume = stock['Volume']
        if volume is None:
            return pd.DataFrame()
            
        dataframe = ta.obv(close, volume) # type: ignore

        if dataframe is None:
            return pd.DataFrame()

        return dataframe


class IndicatorWrapper(Indicator):
    def __init__(self, indicator: List[Indicator]) -> None:
        self.__indicator = indicator

    def __call__(self, stock: Stock) -> pd.DataFrame:
        indicators = {}

        for indicator in self.__indicator:
            indicators[indicator.name] = indicator(stock)

        return pd.DataFrame(indicators)


if __name__ == "__main__":
    AAPL = Stock('AAPL')
    rsi = RSI()(AAPL)
    macd = MACD()(AAPL)
    obv = OBV()(AAPL)
    stoch = StochasticOscillator()(AAPL)

    for indicator in [rsi, macd, obv, stoch]:
        plt.title(indicator.name)
        plt.plot(indicator)
        plt.show()

    indicators = IndicatorWrapper([RSI()])
    indicaotrs_df = indicators(AAPL)
    print(indicaotrs_df.head())

    plt.plot(AAPL['Close'])
    plt.show()

