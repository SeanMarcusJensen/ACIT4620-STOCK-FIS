from pandas_ta import obv
from models import Stock
from .abstraction import Indicator
from pandas import DataFrame

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class OBV(Indicator):
    name = "obv"
    column_names = [name, "OBV_Trend"]

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, stock: Stock) -> DataFrame:
        import pandas as pd
        assert stock['Close'] is not None, 'Close column is missing'
        assert stock['Volume'] is not None, 'Volume column is missing'

        close = stock['Close']
        volume = stock['Volume']
        obv_data = pd.DataFrame(obv(close, volume)) # type: ignore
        obv_data.rename(columns={org: col for org, col in zip(
            obv_data.columns, self.column_names)}, inplace=True)

        prev_obv = None
        for i, row in obv_data.iterrows():
            if prev_obv is None:
                obv_data.loc[i, 'OBV_Trend'] = 0
                prev_obv = row[self.name]
                continue

            if row[self.name] > prev_obv:
                obv_data.loc[i, 'OBV_Trend'] = 1
            elif row[self.name] < prev_obv:
                obv_data.loc[i, 'OBV_Trend'] = -1
            else:
                obv_data.loc[i, 'OBV_Trend'] = 0
            prev_obv = row[self.name]

        return obv_data

    def get_mf(self) -> ctrl.Antecedent:
        obv = ctrl.Antecedent(np.arange(-1e7, 1e7, 1e5), self.name)
        obv['Low'] = fuzz.trapmf(obv.universe, [-1e7, -1e7, -5e6, 0])
        obv['High'] = fuzz.trapmf(obv.universe, [0, 5e6, 1e7, 1e7])
        return obv
