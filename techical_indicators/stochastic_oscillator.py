from pandas_ta import stoch
from models import Stock
from .abstraction import Indicator
from typing import Tuple
import pandas as pd

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

    def __call__(self, stock: Stock, fillna: float | None = None) -> Tuple[ctrl.Antecedent, pd.DataFrame]:
        assert stock['Close'] is not None, 'Close column is missing'
        assert stock['High'] is not None, 'Close column is missing'
        assert stock['Low'] is not None, 'Close column is missing'

        close = stock['Close']
        high = stock['High']
        low = stock['Low']
        data = stoch(high, low, close, **self.__dict__)  # type: ignore
        data.rename(columns={org: col for org, col in zip(
            data.columns, self.column_names)}, inplace=True)
        if fillna is not None:
            data.fillna(fillna, inplace=True)
        else:
            data.dropna(inplace=True)
        return (self.get_mf(data), pd.DataFrame(data[self.name]))

    def get_mf(self, data: pd.DataFrame) -> ctrl.Antecedent:
        so_data = data[self.name]
        low = so_data[so_data < 20]
        mid = so_data[(so_data > 20) & (so_data < 80)]
        high = so_data[so_data > 80]

        so = ctrl.Antecedent(np.arange(0, 101, 1), self.name)
        so['Low'] = fuzz.gaussmf(so.universe, low.mean(), low.std())
        so['Medium'] = fuzz.gaussmf(so.universe, mid.mean(), mid.std())
        so['High'] = fuzz.gaussmf(so.universe, high.mean(), high.std())
        return so
