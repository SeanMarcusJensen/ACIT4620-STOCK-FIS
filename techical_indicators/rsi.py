import numpy as np
import pandas as pd


class RSI:
    def __init__(self, data: pd.DataFrame, period=14):
        self.data = data
        self.period = period
        self.rsi = self.calculate_rsi()

    def calculate_rsi(self) -> np.ndarray:
        delta = np.diff(self.data)
        gain = (delta[delta > 0]).mean()
        loss = (-delta[delta < 0]).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def __call__(self) -> np.ndarray:
        return self.rsi


if __name__ == '__main__':
    data = {}
    rsi = RSI(data, period=14)
