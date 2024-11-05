import pandas_ta as ta
from models import Stock
from .abstraction import Indicator
from pandas import DataFrame

class OBV(Indicator):
    name = "OBV"
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, stock: Stock) -> DataFrame:
        assert stock['Close'] is not None, 'Close column is missing' 
        assert stock['Volume'] is not None, 'Volume column is missing' 

        close = stock['Close']
        volume = stock['Volume']
        dataframe = ta.obv(close, volume) # type: ignore

        return dataframe
