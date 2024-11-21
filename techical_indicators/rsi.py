from pandas_ta import rsi
import pandas as pd
from typing import Tuple
from models import Stock
from .abstraction import Indicator
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class RSI(Indicator):
    name = 'rsi'
    column_names = [name]

    def __init__(self, period: int = 14, magnitude: int = 100) -> None:
        super().__init__()
        self.length = period
        self.scalar = magnitude

    
    def __call__(self, stock: Stock, fillna: float | None = None) -> Tuple[ctrl.Antecedent, pd.DataFrame]:
        """
        STOCK DATA INCOMING:
            60 days worth of 5 min intervals.
            TODO: Do we calculate based on intervals or days in pandas_ta?
            ...is one period 1 day or 1 interval point??.
        """
        import pandas as pd
        close = stock['Close']
        data = pd.DataFrame(rsi(close, **self.__dict__))  # type: ignore
        data.rename(columns={org: col for org, col in zip(
            data.columns, self.column_names)}, inplace=True)
        if fillna is not None:
            data.fillna(fillna, inplace=True)
        else:
            data.dropna(inplace=True)
        return (self.get_mf(data), pd.DataFrame(data[self.name]))
    
    def get_mf(self, data: pd.DataFrame) -> ctrl.Antecedent:
        rsi_data = data[self.name]
        rsi = ctrl.Antecedent(np.arange(0, 101, 0.1), self.name)

        low = rsi_data[rsi_data < 30]
        mid = rsi_data[(rsi_data > 30) & (rsi_data < 70)]
        high = rsi_data[rsi_data > 70]

        rsi['Low'] = fuzz.gaussmf(rsi.universe, low.mean(), low.std())
        rsi['Medium'] = fuzz.gaussmf(rsi.universe, mid.mean(), mid.std())
        rsi['High'] = fuzz.gaussmf(rsi.universe, high.mean(), high.std())
        return rsi

