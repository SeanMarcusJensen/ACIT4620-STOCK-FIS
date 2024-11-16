from pandas_ta import rsi
from models import Stock
from .abstraction import Indicator
from pandas import DataFrame


class RSI(Indicator):
    name = 'RSI'
    column_names = ['RSI']

    def __init__(self, period: int = 14, magnitude: int = 100) -> None:
        super().__init__()
        self.length = period
        self.scalar = magnitude

    def __call__(self, stock: Stock) -> DataFrame:
        import pandas as pd
        close = stock['Close']
        data = rsi(close, **self.__dict__)  # type: ignore
        return pd.DataFrame(data)
