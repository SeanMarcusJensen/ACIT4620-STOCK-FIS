from pandas_ta import rsi
from models import Stock
from .abstraction import Indicator
import numpy as np
from pandas import DataFrame
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class RSI(Indicator):
    name = 'rsi'
    column_names = [name]

    def __init__(self, period: int = 14, magnitude: int = 100) -> None:
        super().__init__()
        self.length = period
        self.scalar = magnitude

    def __call__(self, stock: Stock) -> DataFrame:
        import pandas as pd
        close = stock['Close']
        data = pd.DataFrame(rsi(close, **self.__dict__))  # type: ignore
        data.rename(columns={org: col for org, col in zip(
            data.columns, self.column_names)}, inplace=True)
        return data
    
    def get_mf(self) -> ctrl.Antecedent:
        rsi = ctrl.Antecedent(np.arange(0, 101, 1), self.name)
        rsi['Low'] = fuzz.trimf(rsi.universe, [0, 0, 30])
        rsi['Medium'] = fuzz.trimf(rsi.universe, [30, 50, 70])
        rsi['High'] = fuzz.trimf(rsi.universe, [70, 100, 100])
        return rsi

