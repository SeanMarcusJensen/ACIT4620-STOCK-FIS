import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from utils import download_history
from abc import ABC, abstractmethod


class Stock:
    def __init__(self, ticker, start=None, end=None, period='1d', interval='1m'):
        self.__ticker = ticker
        self.__data = download_history(
            self.__ticker, start=start, end=end, period=period, interval=interval
        )

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


class MACD(Indicator):
    name = 'MACD'

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        super().__init__()
        self.__fast = fast
        self.__slow = slow
        self.__signal = signal

    def __call__(self, stock: Stock) -> pd.DataFrame:
        assert stock['Close'] is not None, 'Close column is missing'
        close = stock['Close']
        if close is None:
            return pd.DataFrame()
        macd = ta.macd(close, fast=self.__fast,
                       slow=self.__slow, signal=self.__signal)
        # MACD, Signal, Histogram
        return macd[['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']]


class StochasticOscillator(Indicator):
    name = 'SO'

    def __init__(self, k: int = 14, d: int = 3, smooth_k: int = 3) -> None:
        super().__init__()
        self.__k = k
        self.__d = d
        self.__smooth_k = smooth_k

    def __call__(self, stock: Stock) -> pd.DataFrame:
        assert stock['Close'] is not None, 'Close column is missing'
        assert stock['High'] is not None, 'High column is missing'
        assert stock['Low'] is not None, 'Low column is missing'
        close = stock['Close']
        high = stock['High']
        low = stock['Low']
        if close is None or high is None or low is None:
            return pd.DataFrame()
        so = ta.stoch(high, low, close, k=self.__k,
                      d=self.__d, smooth_k=self.__smooth_k)
        return so[['STOCHk_14_3_3', 'STOCHd_14_3_3']]  # %K, %D


class OBV(Indicator):
    name = 'OBV'

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, stock: Stock) -> pd.DataFrame:
        assert stock['Close'] is not None, 'Close column is missing'
        assert stock['Volume'] is not None, 'Volume column is missing'
        close = stock['Close']
        volume = stock['Volume']
        if close is None or volume is None:
            return pd.DataFrame()
        obv = ta.obv(close, volume)
        return pd.DataFrame(obv, columns=['OBV'])


# Signal generation using `if` conditions
def generate_rsi_signal(rsi_value):
    if rsi_value < 30:
        return "Buy"
    elif rsi_value > 70:
        return "Sell"
    else:
        return None  # No signal


def generate_macd_signal(macd_value):
    if macd_value > 0:  # MACD above zero
        return "Buy"
    elif macd_value < 0:  # MACD below zero
        return "Sell"
    else:
        return None  # No signal


def generate_so_signal(so_value):
    if so_value < 20:  # Oversold condition
        return "Buy"
    elif so_value > 80:  # Overbought condition
        return "Sell"
    else:
        return None  # No signal


def generate_obv_signal(obv_value, previous_obv_value):
    if obv_value > previous_obv_value:  # OBV is rising
        return "Buy"
    elif obv_value < previous_obv_value:  # OBV is falling
        return "Sell"
    else:
        return None  # No signal


if __name__ == "__main__":
    # Download stock data
    AAPL = Stock('AAPL')

    # RSI Signals
    rsi_indicator = RSI()
    rsi_data = rsi_indicator(AAPL).reindex(AAPL['Close'].index)
    rsi_signals = [generate_rsi_signal(value) for value in rsi_data]

    # MACD Signals
    macd_indicator = MACD()
    macd_data = macd_indicator(AAPL).reindex(AAPL['Close'].index)
    macd_signals = [generate_macd_signal(value)
                    for value in macd_data['MACD_12_26_9']]

    # SO Signals
    so_indicator = StochasticOscillator()
    so_data = so_indicator(AAPL).reindex(AAPL['Close'].index)
    so_signals = [generate_so_signal(value)
                  for value in so_data['STOCHk_14_3_3']]

    # OBV Signals
    obv_indicator = OBV()
    obv_data = obv_indicator(AAPL).reindex(AAPL['Close'].index)
    obv_data['Previous_OBV'] = obv_data['OBV'].shift(1)
    obv_signals = [
        generate_obv_signal(value, previous)
        for value, previous in zip(obv_data['OBV'], obv_data['Previous_OBV'])
    ]

    # Combine signals into a DataFrame
    signal_df = pd.DataFrame({
        'Time': AAPL['Close'].index,
        'RSI_Signal': rsi_signals,
        'MACD_Signal': macd_signals,
        'SO_Signal': so_signals,
        'OBV_Signal': obv_signals
    })

    # Filter signal events
    signal_events = []
    for _, row in signal_df.iterrows():
        if row['RSI_Signal']:
            signal_events.append(
                {'Time': row['Time'], 'Indicator': 'RSI', 'Signal': row['RSI_Signal']})
        if row['MACD_Signal']:
            signal_events.append(
                {'Time': row['Time'], 'Indicator': 'MACD', 'Signal': row['MACD_Signal']})
        if row['SO_Signal']:
            signal_events.append(
                {'Time': row['Time'], 'Indicator': 'SO', 'Signal': row['SO_Signal']})
        if row['OBV_Signal']:
            signal_events.append(
                {'Time': row['Time'], 'Indicator': 'OBV', 'Signal': row['OBV_Signal']})

    # Create signal events DataFrame
    signal_events_df = pd.DataFrame(signal_events)

    # Display the signal events as a table
    print(signal_events_df)

    # Save the signals to a CSV
    signal_events_df.to_csv("signal_events.csv", index=False)

    # Plot the data
    plt.figure(figsize=(14, 14))

    # Plot RSI
    plt.subplot(4, 1, 1)
    plt.plot(rsi_data.index, rsi_data, label='RSI')
    plt.axhline(y=30, color='green', linestyle='--', label='Buy Threshold')
    plt.axhline(y=70, color='red', linestyle='--', label='Sell Threshold')
    plt.legend()
    plt.title("RSI")

    # Plot MACD
    plt.subplot(4, 1, 2)
    plt.plot(macd_data.index, macd_data['MACD_12_26_9'], label='MACD Line')
    plt.plot(macd_data.index, macd_data['MACDs_12_26_9'], label='Signal Line')
    plt.axhline(y=0, color='black', linestyle='--', label='Zero Line')
    plt.legend()
    plt.title("MACD")

    # Plot SO
    plt.subplot(4, 1, 3)
    plt.plot(so_data.index, so_data['STOCHk_14_3_3'], label='SO (%K)')
    plt.plot(so_data.index, so_data['STOCHd_14_3_3'], label='SO (%D)')
    plt.axhline(y=20, color='green', linestyle='--',
                label='Oversold Threshold')
    plt.axhline(y=80, color='red', linestyle='--',
                label='Overbought Threshold')
    plt.legend()
    plt.title("Stochastic Oscillator")

    # Plot OBV
    plt.subplot(4, 1, 4)
    plt.plot(obv_data.index, obv_data['OBV'], label='OBV')
    plt.legend()
    plt.title("On-Balance Volume (OBV)")

    plt.tight_layout()
    plt.show()

    # Plot stock price movement in a separate window
    plt.figure(figsize=(14, 7))
    plt.plot(AAPL['Close'].index, AAPL['Close'],
             label='Stock Price', color='blue')
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Stock Price Movement")
    plt.legend()
    plt.show()
