from pandas_ta import macd
from models import Stock
from .abstraction import Indicator
from pandas import DataFrame
from typing import Tuple
import pandas as pd

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

    def __call__(self, stock: Stock) -> Tuple[ctrl.Antecedent, DataFrame]:
        """
        Produces 3 time series: MACD Line, Signal Line, and MACD Histogram.
        """
        import pandas as pd
        close = stock['Close']
        data = pd.DataFrame(macd(close, **self.__dict__)) # type: ignore
        data.rename(columns={org: col for org, col in zip(
            data.columns, self.column_names)}, inplace=True)

        return (self.get_mf(data), data)

    def get_mf(self, data: pd.DataFrame) -> ctrl.Antecedent:
        macd_data = (data[self.name] - data['macds'])
        low = macd_data[macd_data < 0]
        high = macd_data[macd_data > 0]

        macd = ctrl.Antecedent(np.linspace(macd_data.min(), macd_data.max(), len(macd_data)), self.name)
        macd['Low'] = fuzz.gaussmf(macd.universe, low.mean(), low.std())
        macd['High'] = fuzz.gaussmf(macd.universe, high.mean(), high.std())

        import matplotlib.pyplot as plt
        macd.view()
        plt.show()

        return macd
