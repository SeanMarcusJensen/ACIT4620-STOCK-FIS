from pandas_ta import stoch
from models import Stock
from .abstraction import Indicator
from pandas import DataFrame

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class StochasticOscillator(Indicator):
    name = "so"
    column_names = [name, 'STOCH_d', 'STOCH_s']

    def __init__(self, fast_k: int = 14, slow_d: int = 3, slow_k: int = 3) -> None:
        super().__init__()
        self.k = fast_k
        self.d = slow_d
        self.smooth_k = slow_k

    def __call__(self, stock: Stock) -> DataFrame:
        import pandas as pd
        assert stock['Close'] is not None, 'Close column is missing'
        assert stock['High'] is not None, 'Close column is missing'
        assert stock['Low'] is not None, 'Close column is missing'

        close = stock['Close']
        high = stock['High']
        low = stock['Low']
        data = pd.DataFrame(stoch(high, low, close, **self.__dict__)) # type: ignore
        data.rename(columns={org: col for org, col in zip(
            data.columns, self.column_names)}, inplace=True)
        return data

    def get_mf(self) -> ctrl.Antecedent:
        so = ctrl.Antecedent(np.arange(0, 101, 1), self.name)
        so['Low'] = fuzz.trimf(so.universe, [0, 0, 20])
        so['Medium'] = fuzz.trimf(so.universe, [20, 50, 80])
        so['High'] = fuzz.trimf(so.universe, [80, 100, 100])
        return so
