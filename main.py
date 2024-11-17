from models import Stock
from utils import DateChunker
from techical_indicators import RSI, MACD, OBV, StochasticOscillator, IndicatorWrapper
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import pandas as pd
import matplotlib.pyplot as plt

def create_system(df: pd.DataFrame):
    import numpy as np

    # Might add to 'Indicator' class
    min, max = df['RSI'].min(), df['RSI'].max()
    rsi = ctrl.Antecedent(label='RSI', universe=np.arange(min, max, like=np.array(df['RSI'])))
    rsi['Low'] = fuzz.trimf(rsi.universe, [0, 0, 30])
    rsi['Medium'] = fuzz.trimf(rsi.universe, [30, 50, 70])
    rsi['High'] = fuzz.trimf(rsi.universe, [70, 100, 100])
    rsi.view()
    plt.show()

    min, max = df['STOCH_k'].min(), df['STOCH_k'].max()
    so = ctrl.Antecedent(label='Stochastic Oscillator', universe=np.arange(min, max, like=np.array(df['STOCH_k'])))
    so['Low'] = fuzz.trimf(so.universe, [0, 0, 20])
    so['Medium'] = fuzz.trimf(so.universe, [20, 50, 80])
    so['High'] = fuzz.trimf(so.universe, [80, 100, 100])
    so.view()
    plt.show()

    min, max = df['OBV_Trend'].min(), df['OBV_Trend'].max()
    obv_trend = ctrl.Antecedent(label='OBV Trend', universe=np.arange(min, max, like=np.array(df['OBV_Trend'])))
    obv_trend['Low'] = fuzz.trimf(obv_trend.universe, [-1., -1., 0.])
    obv_trend['High'] = fuzz.trimf(obv_trend.universe, [0., 1., 1.])
    obv_trend.view()
    plt.show()

    ctrl.Consequent(label='RSI Signal', universe=['Buy', 'Hold', 'Sell'])



if __name__ == "__main__":
    STOCK = Stock('AAPL')
    indicators = [RSI(), MACD(), OBV(), StochasticOscillator()]
    indicator = IndicatorWrapper(indicators, fillna=0)

    data = indicator(STOCK)
    chunker = DateChunker(data)

    # for chunk in iter(chunker):
    #     last_row = None
    #     for index, row in chunk.iterrows():
    #         if last_row is None:
    #             last_row = row
    #             continue
    #         
    #         print(row)
    #         if row['RSI'] > 70 and last_row['RSI'] < 70:
    #             print(f"RSI Overbought on {index}")

    #         last_row = row
    #     break

    print(chunker.get_date('2024-11-01').head())
    create_system(data)

