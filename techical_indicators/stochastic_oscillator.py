import pandas_ta as ta
from models import Stock
from .abstraction import Indicator
from pandas import DataFrame

class StochasticOscillator(Indicator):
    name = "Stochastic Oscillator"
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, stock: Stock) -> DataFrame:
        assert stock['Close'] is not None, 'Close column is missing' 
        assert stock['High'] is not None, 'Close column is missing' 
        assert stock['Low'] is not None, 'Close column is missing' 

        close = stock['Close']
        high = stock['High']
        low = stock['Low']
        dataframe = ta.stoch(high, low, close) # type: ignore
        return dataframe
