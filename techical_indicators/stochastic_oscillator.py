from pandas_ta import stoch
from models import Stock
from .abstraction import Indicator
from pandas import DataFrame


class StochasticOscillator(Indicator):
    name = "Stochastic Oscillator"
    column_names = ['STOCH_k', 'STOCH_d', 'STOCH_s']

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
        data = stoch(high, low, close, **self.__dict__)  # type: ignore
        return pd.DataFrame(data)
