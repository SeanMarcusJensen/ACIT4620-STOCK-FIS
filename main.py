import numpy as np
import skfuzzy as fuzz
from models import Stock
from utils import DateChunker
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
from techical_indicators import RSI, MACD, OBV, StochasticOscillator, IndicatorWrapper

def create_system():

    rsi = ctrl.Antecedent(np.arange(0, 101, 1), 'RSI')
    rsi['Low'] = fuzz.trimf(rsi.universe, [0, 0, 30])
    rsi['Medium'] = fuzz.trimf(rsi.universe, [30, 50, 70])
    rsi['High'] = fuzz.trimf(rsi.universe, [70, 100, 100])

    macd = ctrl.Antecedent(np.arange(-5, 5, 0.1), 'MACD')
    macd['Low'] = fuzz.trapmf(macd.universe, [-5.0, -5.0, -1.0, 0.0])
    macd['Medium'] = fuzz.trimf(macd.universe, [-1.0, 0.0, 1.0])
    macd['High'] = fuzz.trapmf(macd.universe, [0.0, 1.0, 5.0, 5.0])

    so = ctrl.Antecedent(np.arange(0, 101, 1), 'SO')
    so['Low'] = fuzz.trimf(so.universe, [0, 0, 20])
    so['Medium'] = fuzz.trimf(so.universe, [20, 50, 80])
    so['High'] = fuzz.trimf(so.universe, [80, 100, 100])

    obv = ctrl.Antecedent(np.arange(-1e7, 1e7, 1e5), 'OBV')
    obv['Low'] = fuzz.trapmf(obv.universe, [-1e7, -1e7, -5e6, 0])
    obv['High'] = fuzz.trapmf(obv.universe, [0, 5e6, 1e7, 1e7])

    action = ctrl.Consequent(np.arange(0, 101, 1), 'Action')
    action['Sell'] = fuzz.trapmf(action.universe, [0, 0, 30, 50])
    action['Hold'] = fuzz.trimf(action.universe, [30, 50, 70])
    action['Buy'] = fuzz.trapmf(action.universe, [50, 70, 100, 100])

    rules = [
            ctrl.Rule(macd['Low']   & rsi['Low']    & so['Low']     & obv['Low'],   action['Buy']),
            ctrl.Rule(macd['Low']   & rsi['Low']    & so['Low']     & obv['High'],  action['Hold']),
            ctrl.Rule(macd['High']  & rsi['Low']    & so['Low']     & obv['Low'],   action['Sell']),
            ctrl.Rule(macd['High']  & rsi['Low']    & so['Low']     & obv['High'],  action['Buy']),
            ctrl.Rule(macd['Low']   & rsi['Low']    & so['Medium']  & obv['Low'],   action['Hold']),
            ctrl.Rule(macd['Low']   & rsi['Low']    & so['Medium']  & obv['High'],  action['Hold']),
            ctrl.Rule(macd['High']  & rsi['Low']    & so['Medium']  & obv['High'],  action['Sell']),
            ctrl.Rule(macd['Low']   & rsi['Medium'] & so['Medium']  & obv['Low'],   action['Sell']),
            ctrl.Rule(macd['Low']   & rsi['Medium'] & so['Low']     & obv['High'],  action['Buy']),
            ctrl.Rule(macd['High']  & rsi['Medium'] & so['High']    & obv['Low'],   action['Buy']),
            ctrl.Rule(macd['High']  & rsi['Medium'] & so['Low']     & obv['High'],  action['Buy']),
            ctrl.Rule(macd['Low']   & rsi['Low']    & so['High']    & obv['High'],  action['Sell']),
    ]

    system = ctrl.ControlSystem(rules)
    simulator = ctrl.ControlSystemSimulation(system)
    return simulator


class StockIndicator:
    def __init__(self, indicators: list) -> None:
        self.indicators = indicators
        self.columns = [indicator.name for indicator in indicators]
    
    def action(self, series) -> str:
        assert len(series) >= len(self.columns)
        assert series.columns == self.columns

        for index, row in series.iterrows():
            pass

        return ''


if __name__ == "__main__":
    STOCK = Stock('AAPL')
    indicators = [RSI(), MACD(), OBV(), StochasticOscillator()]
    indicator = IndicatorWrapper(indicators, fillna=0)

    data = indicator(STOCK)
    chunker = DateChunker(data)
    simulator = create_system()

    for chunk in iter(chunker):
        decisions = []
        for index, row in chunk.iterrows():
            simulator.input['RSI'] = row['RSI']
            simulator.input['OBV'] = row['OBV']
            simulator.input['MACD'] = row['MACD']
            simulator.input['SO'] = row['SO']
            simulator.compute()
            decisions.append(simulator.output.get('Action', np.nan))

        chunk['Decicion'] = decisions
        chunk['Action'] = chunk['Decicion'].apply(lambda x: 'Buy' if x > 50 else 'Sell' if x < 50 else 'Hold')

    
    data = chunker.get_data()

    plt.plot(data['Close'], label='Close Price')

    # Create a subset of the data with only the relevant columns for plotting
    plot_data = data[['Close', 'Action']].copy()

    # Define colors for buy, hold, and sell signals
    colors = {'Buy': 'green', 'Sell': 'red', 'Hold': 'orange'}

    # Plot the closing prices
    plt.figure(figsize=(14, 8))
    plt.plot(plot_data.index, plot_data['Close'], label='Close Price', color='blue', alpha=0.6)
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('AAPL Closing Price with Buy/Hold/Sell Signals')

    # Annotate Buy, Hold, and Sell signals
    for idx, row in plot_data.iterrows():
        if row['Action'] == 'Buy':
            plt.annotate('Buy', (idx, row['Close']), textcoords="offset points", xytext=(0, 10),
                         ha='center', color=colors['Buy'], fontsize=10, fontweight='bold')
        elif row['Action'] == 'Sell':
            plt.annotate('Sell', (idx, row['Close']), textcoords="offset points", xytext=(0, -15),
                         ha='center', color=colors['Sell'], fontsize=10, fontweight='bold')

    # Adding the legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label='Buy', markerfacecolor=colors['Buy'], markersize=10),
               plt.Line2D([0], [0], marker='o', color='w', label='Sell', markerfacecolor=colors['Sell'], markersize=10)]
    plt.legend(handles=handles, loc='upper left')

    plt.show()
