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
    rsi.view()

    macd = ctrl.Antecedent(np.arange(-5, 5, 0.1), 'MACD')
    macd['Low'] = fuzz.trapmf(macd.universe, [-5.0, -5.0, -1.0, 0.0])
    macd['Medium'] = fuzz.trimf(macd.universe, [-1.0, 0.0, 1.0])
    macd['High'] = fuzz.trapmf(macd.universe, [0.0, 1.0, 5.0, 5.0])
    macd.view()

    so = ctrl.Antecedent(np.arange(0, 101, 1), 'SO')
    so['Low'] = fuzz.trimf(so.universe, [0, 0, 20])
    so['Medium'] = fuzz.trimf(so.universe, [20, 50, 80])
    so['High'] = fuzz.trimf(so.universe, [80, 100, 100])
    so.view()

    obv = ctrl.Antecedent(np.arange(-1e7, 1e7, 1e5), 'OBV')
    obv['Low'] = fuzz.trapmf(obv.universe, [-1e7, -1e7, -5e6, 0])
    obv['High'] = fuzz.trapmf(obv.universe, [0, 5e6, 1e7, 1e7])
    obv.view()

    action = ctrl.Consequent(np.arange(0, 101, 1), 'Action')
    action['Sell'] = fuzz.trapmf(action.universe, [0, 0, 30, 50])
    action['Hold'] = fuzz.trimf(action.universe, [30, 50, 70])
    action['Buy'] = fuzz.trapmf(action.universe, [50, 70, 100, 100])
    action.view()

    plt.show()
    rules = [
            ctrl.Rule(macd['High']   & rsi['Low']    & so['Low']     & obv['High'],   action['Buy']),
            ctrl.Rule(macd['Low']   & rsi['High']    & so['High']     & obv['Low'],  action['Buy']),
            ctrl.Rule(macd['High']  & rsi['Medium']    & so['Medium']     & obv['High'],   action['Buy']),
            ctrl.Rule(rsi['Low']    & so['Low']     & obv['High'],  action['Buy']),
            ctrl.Rule(macd['Low']   & rsi['Medium']    & so['High']  & obv['Low'],   action['Sell']),
            ctrl.Rule(rsi['High']    & so['High']  & obv['Low'],  action['Sell']),
            ctrl.Rule(macd['Low']  & rsi['High']    & so['High'],  action['Sell']),
            ctrl.Rule(macd['Low']   & rsi['Medium'] & so['Medium'],   action['Hold']),
            ctrl.Rule(macd['High']   & rsi['Medium'] & so['Medium']     & obv['Low'],  action['Hold']),
    ]

    system = ctrl.ControlSystem(rules)
    simulator = ctrl.ControlSystemSimulation(system)
    return simulator, action


if __name__ == "__main__":
    STOCK = Stock('PLTR')
    indicators = [RSI(), MACD(), OBV(), StochasticOscillator()]
    indicator = IndicatorWrapper(indicators, fillna=0)

    data = indicator(STOCK)
    chunker = DateChunker(data)
    simulator, action = create_system()

    class Wallet:
        def __init__(self, value):
            self.starting_value = value
            self.value = value
            self.last_action: str | None = None
            self.n_shares = 0
            self.buy_price = 0.0
            self.history = []
            self.comission = 0.0

            self.successful_trades = 0
            self.trades = 0
            self.biggest_gain = 0.0

        def buy(self, price, index):
            if self.last_action == 'Sell' or self.last_action is None:
                self.trades += 1
                # Buy as many shares
                # Buy for 2-5% of wallet each trade
                self.buy_price = price
                self.n_shares = self.value // price
                self.value = self.value % price
                self.value -= self.comission
                self.last_action = 'Buy'
                self.history.append((index, price, 'Buy'))

        def sell(self, price, index):
            if self.last_action == 'Buy':
                self.trades += 1
                gain = (price * self.n_shares)
                self.biggest_gain = max(self.biggest_gain, gain)
                self.value += gain
                self.value -= self.comission
                self.n_shares = 0
                self.last_action = 'Sell'
                self.buy_price = 0.0
                self.history.append((index, price, 'Sell'))

                if price > self.buy_price:
                    self.successful_trades += 1

        def profits(self):
            return (self.value + (self.n_shares * self.buy_price)) - self.starting_value

        def __str__(self):
            trades = (self.successful_trades / self.trades) * 100
            return f"Wallet: [{self.value}, @profit={self.profits()}], Trades: [{self.successful_trades}/{self.trades}({trades}%)], Shares: [n:{self.n_shares}, @{self.buy_price}], Last Action: {self.last_action}, Biggest Winner: {self.biggest_gain}"

    wallet = Wallet(10000)

    for chunk in iter(chunker):
        decisions = []
        for index, row in chunk.iterrows():
            simulator.input['RSI'] = row['RSI']
            simulator.input['OBV'] = row['OBV']
            simulator.input['MACD'] = (row['MACD'] - row['MACDs'])
            simulator.input['SO'] = row['SO']
            simulator.compute()
            current_action = simulator.output.get('Action', np.nan)
            decisions.append(current_action)
        chunk['Decicion'] = decisions
        chunk['Action'] = chunk['Decicion'].apply(lambda x: 'Buy' if x > 50 else 'Sell' if x < 50 else 'Hold')

    for chunk in iter(chunker):
        for index, row in chunk.iterrows():
            if row['Action'] == 'Buy':
                wallet.buy(row['Close'], index)
            elif row['Action'] == 'Sell':
                wallet.sell(row['Close'], index)

        print(wallet.__str__())
        plt.plot(chunk['Close'])
        buys = [x for x in wallet.history if x[2] == 'Buy']
        sells = [x for x in wallet.history if x[2] == 'Sell']
        plt.plot(np.array([x[0] for x in buys]), np.array([x[1] for x in buys]), 'g^', markersize=10, label='Buy', color='green')
        plt.plot(np.array([x[0] for x in sells]), np.array([x[1] for x in sells]), 'g^', markersize=10, label='Sell', color='red')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.xticks(rotation=45, ha='right')
        plt.show()

    
    print(wallet.__str__())
    print(f"Start: {data.iloc[0]['Close']} - End: {data.iloc[-1]['Close']}")

    plt.plot(data['Close'])
    buys = [x for x in wallet.history if x[2] == 'Buy']
    sells = [x for x in wallet.history if x[2] == 'Sell']
    plt.plot(np.array([x[0] for x in buys]), np.array([x[1] for x in buys]), 'g^', markersize=10, label='Buy', color='green')
    plt.plot(np.array([x[0] for x in sells]), np.array([x[1] for x in sells]), 'g^', markersize=10, label='Sell', color='red')
    plt.show()


    #     data = chunk

    #     plt.plot(data['Close'], label='Close Price')

    #     # Create a subset of the data with only the relevant columns for plotting
    #     plot_data = data[['Close', 'Action']].copy()

    #     # Define colors for buy, hold, and sell signals
    #     colors = {'Buy': 'green', 'Sell': 'red', 'Hold': 'orange'}

    #     # Plot the closing prices
    #     plt.figure(figsize=(14, 8))
    #     plt.plot(plot_data.index, plot_data['Close'], label='Close Price', color='blue', alpha=0.6)
    #     plt.xlabel('Date')
    #     plt.ylabel('Closing Price')
    #     plt.title('AAPL Closing Price with Buy/Hold/Sell Signals')

    #     # Annotate Buy, Hold, and Sell signals
    #     for idx, row in plot_data.iterrows():
    #         if row['Action'] == 'Buy':
    #             plt.annotate('Buy', (idx, row['Close']), textcoords="offset points", xytext=(0, 10),
    #                          ha='center', color=colors['Buy'], fontsize=10, fontweight='bold')
    #         elif row['Action'] == 'Sell':
    #             plt.annotate('Sell', (idx, row['Close']), textcoords="offset points", xytext=(0, -15),
    #                          ha='center', color=colors['Sell'], fontsize=10, fontweight='bold')

    #     # Adding the legend
    #     handles = [plt.Line2D([0], [0], marker='o', color='w', label='Buy', markerfacecolor=colors['Buy'], markersize=10),
    #                plt.Line2D([0], [0], marker='o', color='w', label='Sell', markerfacecolor=colors['Sell'], markersize=10)]
    #     plt.legend(handles=handles, loc='upper left')

    #     plt.show()
