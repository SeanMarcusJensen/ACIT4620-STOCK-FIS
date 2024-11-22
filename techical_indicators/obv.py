from pandas_ta import obv
import pandas as pd
from typing import Tuple
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

    def __call__(self, stock: Stock, fillna: float | None = None) -> Tuple[ctrl.Antecedent, DataFrame]:
        assert stock['Close'] is not None, 'Close column is missing'
        assert stock['Volume'] is not None, 'Volume column is missing'

        close = stock['Close']
        volume = stock['Volume']
        obv_data = pd.DataFrame(obv(close, volume)) # type: ignore

        obv_data.rename(columns={org: col for org, col in zip(
            obv_data.columns, self.column_names)}, inplace=True)

        if fillna is not None:
            obv_data.fillna(fillna, inplace=True)
        else:
            obv_data.dropna(inplace=True)

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
                obv_data.loc[i, 'OBV_Trend'] = prev_obv

            prev_obv = row[self.name]

        return (self.get_mf(obv_data), pd.DataFrame(obv_data[self.name]))

    def get_mf(self, data: pd.DataFrame) -> ctrl.Antecedent:
        obv_data = data[self.name]
        low = obv_data[obv_data <= 0]
        high = obv_data[obv_data > 0]

        obv = ctrl.Antecedent(np.linspace(data[self.name].min(), data[self.name].max(), num=len(obv_data)), self.name)
        obv['Low'] = fuzz.gaussmf(obv.universe, low.mean(), low.std())
        obv['High'] = fuzz.gaussmf(obv.universe, high.mean(), high.std())

        import matplotlib.pyplot as plt
        obv.view()
        plt.title("OBV")
        plt.show()

        return obv
