from pandas_ta import obv
from models import Stock
from .abstraction import Indicator
from pandas import DataFrame


class OBV(Indicator):
    name = "OBV"
    column_names = ["OBV"]

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, stock: Stock) -> DataFrame:
        import pandas as pd
        assert stock['Close'] is not None, 'Close column is missing'
        assert stock['Volume'] is not None, 'Volume column is missing'

        close = stock['Close']
        volume = stock['Volume']
        data = obv(close, volume)  # type: ignore

        return pd.DataFrame(data)
