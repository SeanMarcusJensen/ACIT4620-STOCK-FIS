from pandas_ta import obv
from models import Stock
from .abstraction import Indicator
from pandas import DataFrame


class OBV(Indicator):
    name = "OBV"
    column_names = ["OBV", "OBV_Trend"]

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, stock: Stock) -> DataFrame:
        import pandas as pd
        assert stock['Close'] is not None, 'Close column is missing'
        assert stock['Volume'] is not None, 'Volume column is missing'

        close = stock['Close']
        volume = stock['Volume']
        obv_data = pd.DataFrame(obv(close, volume)) # type: ignore

        prev_obv = None
        for i, row in obv_data.iterrows():
            if prev_obv is None:
                obv_data.loc[i, 'OBV_Trend'] = 0
                prev_obv = row['OBV']
                continue

            if row['OBV'] > prev_obv:
                obv_data.loc[i, 'OBV_Trend'] = 1
            elif row['OBV'] < prev_obv:
                obv_data.loc[i, 'OBV_Trend'] = -1
            else:
                obv_data.loc[i, 'OBV_Trend'] = 0
            prev_obv = row['OBV']

        return obv_data
