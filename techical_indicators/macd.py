from pandas_ta import macd
from models import Stock
from .abstraction import Indicator
from pandas import DataFrame

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class MACD(Indicator):
    name = 'macd'
    column_names = ['macd', 'macdh', 'macds']

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        super().__init__()
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def __call__(self, stock: Stock) -> DataFrame:
        """
        Produces 3 time series: MACD Line, Signal Line, and MACD Histogram.
        """
        import pandas as pd
        close = stock['Close']
        data = pd.DataFrame(macd(close, **self.__dict__)) # type: ignore
        data.rename(columns={org: col for org, col in zip(
            data.columns, self.column_names)}, inplace=True)
        return data

    def get_mf(self) -> ctrl.Antecedent:
        macd = ctrl.Antecedent(np.arange(-5, 5, 0.1), self.name)
        macd['Low'] = fuzz.trapmf(macd.universe, [-5.0, -5.0, -1.0, 0.0])
        macd['Medium'] = fuzz.trimf(macd.universe, [-1.0, 0.0, 1.0])
        macd['High'] = fuzz.trapmf(macd.universe, [0.0, 1.0, 5.0, 5.0])
        return macd
