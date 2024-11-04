import pandas as pd
import pandas_ta as ta
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from utils import download_history
from abc import ABC, abstractmethod
from typing import List


class Stock:
    def __init__(self, ticker, start=None, end=None, period='1d', interval='1m'):
        self.__ticker = ticker
        self.__data = download_history(self.__ticker,
                                     start=start, end=end,
                                     period=period, interval=interval)

    def plot(self, key):
        data = self.__data[key]
        if data is None:
            return

        plt.plot(data)
        plt.show()

    def __getitem__(self, key):
        try:
            return self.__data[key] 
        except KeyError:
            return None


class Indicator(ABC):
    name: str
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, stock: Stock) -> pd.DataFrame:
        raise NotImplementedError


class RSI(Indicator):
    name = 'RSI'
    def __init__(self, period: int = 14) -> None:
        super().__init__()
        self.__period = period

    def __call__(self, stock: Stock) -> pd.DataFrame:
        assert stock['Close'] is not None, 'Close column is missing' 
        close = stock['Close']
        if close is None:
            return pd.DataFrame()
        return ta.rsi(close, length=self.__period)


class FuzzyRSISignal:
    def __init__(self):
        # Define fuzzy variables for RSI
        self.rsi = ctrl.Antecedent(np.arange(0, 101, 1), 'rsi')
        self.signal = ctrl.Consequent(np.arange(0, 2, 1), 'signal')

        # Define triangular membership functions for "low" and "high" RSI
        self.rsi['low'] = fuzz.trimf(self.rsi.universe, [0, 30, 30])
        self.rsi['high'] = fuzz.trimf(self.rsi.universe, [70, 70, 100])

        # Define triangular membership functions for the output signal
        self.signal['buy'] = fuzz.trimf(self.signal.universe, [0, 0, 1])
        self.signal['sell'] = fuzz.trimf(self.signal.universe, [1, 1, 1])

        # Define fuzzy rules
        rule1 = ctrl.Rule(self.rsi['low'], self.signal['buy'])
        rule2 = ctrl.Rule(self.rsi['high'], self.signal['sell'])

        # Control system for the rules
        self.signal_ctrl = ctrl.ControlSystem([rule1, rule2])
        self.signal_sim = ctrl.ControlSystemSimulation(self.signal_ctrl)

    def generate_signal(self, rsi_value):
        # Set the input value
        self.signal_sim.input['rsi'] = rsi_value
        
        # Debug: print the input value for each call
        print(f"Input RSI: {rsi_value}")

        # Compute the result
        try:
            self.signal_sim.compute()
            output_signal = self.signal_sim.output['signal']
            
            # Debug: print the output signal
            print(f"Generated Signal: {output_signal}")
            
            return output_signal
        except Exception as e:
            print(f"Error during fuzzy computation: {e}")
            return np.nan  # Return NaN if there's an error


if __name__ == "__main__":
    # Download stock data
    AAPL = Stock('AAPL')
    rsi_indicator = RSI()
    rsi_data = rsi_indicator(AAPL)

    # Initialize fuzzy RSI signal generator
    fuzzy_rsi_signal = FuzzyRSISignal()

    # Calculate signals
    signals = []
    for rsi_value in rsi_data:
        if np.isnan(rsi_value):
            signals.append(np.nan)
        else:
            signals.append(fuzzy_rsi_signal.generate_signal(rsi_value))

    # Convert signals to a DataFrame for easy analysis
    signals_df = pd.DataFrame({
        'RSI': rsi_data,
        'Signal': signals
    })

    # Plot RSI and generated signals
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(signals_df['RSI'], label='RSI')
    plt.axhline(y=30, color='green', linestyle='--', label='Buy Threshold')
    plt.axhline(y=70, color='red', linestyle='--', label='Sell Threshold')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(signals_df['Signal'], label='Signal (0=Buy, 1=Sell)', marker='o')
    plt.legend()
    plt.show()

