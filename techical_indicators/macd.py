import pandas_ta as ta
from models import Stock
from .abstraction import Indicator
from pandas import DataFrame


class MACD(Indicator):
    name = 'MACD'

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, stock: Stock) -> DataFrame:
        """
        Produces 3 time series: MACD Line, Signal Line, and MACD Histogram.
        """
        close = stock['Close']
        dataframe = ta.macd(close) # type: ignore
        return dataframe
