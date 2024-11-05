import pandas_ta as ta
from models import Stock
from .abstraction import Indicator
from pandas import DataFrame

class RSI(Indicator):
    name = 'RSI'
    def __init__(self, period: int = 14) -> None:
        super().__init__()
        self.__period = period
        print("hello world")

    def __call__(self, stock: Stock) -> DataFrame:
        close = stock['Close']
        return ta.rsi(close, offset=self.__period) # type: ignore 
