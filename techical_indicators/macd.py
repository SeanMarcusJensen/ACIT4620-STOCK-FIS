from pandas_ta import macd
from models import Stock
from .abstraction import Indicator
from pandas import DataFrame


class MACD(Indicator):
    name = 'MACD'
    column_names = ['MACD', 'MACDh', 'MACDs']

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        super().__init__()
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def __call__(self, stock: Stock) -> DataFrame:
        """
        Produces 3 time series: MACD Line, Signal Line, and MACD Histogram.
        """
        import pandas as pd
        close = stock['Close']
        data = macd(close, **self.__dict__)
        return pd.DataFrame(data)
