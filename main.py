import numpy as np
from pandas.core.interchange.dataframe_protocol import enum
import skfuzzy as fuzz
import pandas as pd
from typing import List, Dict
from techical_indicators.abstraction import Indicator
from models import Stock
from utils import DateChunker
import pandas as pd
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
from techical_indicators import RSI, MACD, OBV, StochasticOscillator


class Signal(enum.Enum):
    STRONGSELL = 'strong_sell'
    SELL = 'sell'
    HOLD = 'hold'
    BUY = 'buy'
    STRONGBUY = 'strong_buy'


class Wallet:
    def __init__(self, value):
        self.starting_value = value
        self.value = value
        self.last_action: Signal | None = None
        self.n_shares = 0
        self.buy_price = 0.0
        self.history = []
        self.comission = 0.0
        self.successful_trades = 0
        self.trades = 0
        self.biggest_gain = 0.0

    def act(self, signal: Signal, price, index) -> float:
        match signal:
            case Signal.STRONGBUY:
                self.buy(price, index)
            case Signal.BUY:
                self.buy(price, index, 1.)
            case Signal.SELL:
                self.sell(price, index, 1.)
            case Signal.STRONGSELL:
                self.sell(price, index)

        return self.profits()

    def buy(self, price, index, pct = 1.0):
        buy_n_shares = (self.value * pct) // price
        if buy_n_shares < 1:
            return

        self.n_shares = buy_n_shares
        self.trades += 1
        self.buy_price = price
        self.value -= (price * buy_n_shares)
        self.value -= self.comission
        self.last_action = Signal.BUY
        self.history.append((index, price, Signal.BUY))

    def sell(self, price, index, pct = 1.0):
        import math
        sell_n_shares = math.floor(self.n_shares * pct)

        if sell_n_shares < 1:
            return

        self.trades += 1
        gain = (price * sell_n_shares)
        self.biggest_gain = max(self.biggest_gain, gain)
        self.value += gain
        self.value -= self.comission
        self.n_shares -= sell_n_shares
        self.last_action = Signal.SELL
        self.history.append((index, price, Signal.SELL))

        if price > self.buy_price:
            self.successful_trades += 1

    def profits(self):
        return (self.value + (self.n_shares * self.buy_price)) - self.starting_value

    def __str__(self):
        trades = (self.successful_trades / self.trades) * 100 if (self.trades > 0 and self.successful_trades > 0) else 0
        return f"Wallet: [{self.value}, @profit={self.profits()}], Trades: [{self.successful_trades}/{self.trades}({trades}%)], Shares: [n:{self.n_shares}, @{self.buy_price}], Last Action: {self.last_action}, Biggest Winner: {self.biggest_gain}"


class Predictor:
    def __init__(self, sets: List[ctrl.Antecedent]):
        self.sets: Dict[str, ctrl.Antecedent] = {fs.label: fs for fs in sets}
        self.action = ctrl.Consequent(np.arange(0, 31, 1), 'action')
        self.action['Sell'] = fuzz.trimf(self.action.universe, [0, 5, 10])
        self.action['Hold'] = fuzz.trimf(self.action.universe, [10, 15, 20])
        self.action['Buy'] = fuzz.trimf(self.action.universe, [20, 25, 30])

        self.simulator: ctrl.ControlSystemSimulation = self.__construct_system()

    def get_signal(self, series: pd.Series, plot_signal: bool = False) -> Signal:
        self.simulator.inputs({name: series[name] for name in self.sets.keys()})
        self.simulator.compute()
        output = self.simulator.output.get('action', np.nan)
        if plot_signal:
            self.simulator.compute()
            self.action.view(sim=self.simulator)
            plt.show()
        return self.__get_output_signal(output)

    def __construct_system(self):

        rules = [
                ctrl.Rule(self['macd']['High']   & self['rsi']['Low']    & self['so']['Low']     & self['obv']['High'],   self.action['Buy']),
                ctrl.Rule(self['macd']['Low']   & self['rsi']['High']    & self['so']['High']     & self['obv']['Low'],  self.action['Buy']),
                ctrl.Rule(self['macd']['High']  & self['rsi']['Medium']    & self['so']['Medium']     & self['obv']['High'],   self.action['Buy']),
                ctrl.Rule(self['rsi']['Low']    & self['so']['Low']     & self['obv']['High'],  self.action['Buy']),
                ctrl.Rule(self['macd']['Low']   & self['rsi']['Medium']    & self['so']['High']  & self['obv']['Low'],   self.action['Sell']),
                ctrl.Rule(self['rsi']['High']    & self['so']['High']  & self['obv']['Low'],  self.action['Sell']),
                ctrl.Rule(self['macd']['Low']  & self['rsi']['High']    & self['so']['High'],  self.action['Sell']),
                ctrl.Rule(self['macd']['Low']   & self['rsi']['Medium'] & self['so']['Medium'],   self.action['Hold']),
                ctrl.Rule(self['macd']['High']   & self['rsi']['Medium'] & self['so']['Medium']     & self['obv']['Low'],  self.action['Hold']),
        ]

        system = ctrl.ControlSystem(rules)
        return ctrl.ControlSystemSimulation(system)

    def __get_output_signal(self, value: float) -> Signal:
        if value <= 10:
            return Signal.SELL
        elif value >= 20:
            return Signal.BUY
        else:
            return Signal.HOLD

    def __getitem__(self, key) -> ctrl.Antecedent:
        return self.sets[key]


class System:
    def __init__(self, indicators: List[Indicator], fillna: float | None = None):
        self.indicators = indicators
        self.fillna = fillna

    def __call__(self, stock: Stock, print_mf: bool = False) -> pd.DataFrame:
        import time
        start = time.time()

        wallet = Wallet(10000)
        mfs, indicators = zip(*[i(stock, self.fillna) for i in self.indicators])
        data = pd.concat([stock.get_data(), *[i for i in indicators]], axis=1)
        if self.fillna is not None:
            data.fillna(self.fillna, inplace=True)
        else:
            data.dropna(inplace=True)

        if print_mf:
            for mf in mfs:
                mf.view()
            plt.show()

        predictor = Predictor([mf for mf in mfs])

        for index, row in data.iterrows():
            action = predictor.get_signal(row, plot_signal=print_mf)
            data.loc[index, 'action'] = action.value

            returns = wallet.act(action, row['Close'], index)
            data.loc[index, 'returns'] = returns

        end = time.time()
        delta = end - start
        print(f"System took {delta} seconds to complete")
        print(wallet)
        print(data.head())

        return data


if __name__ == "__main__":
    import os
    OUTPUT_FOLDER = 'output'
    STOCKS = {
            'AAPL': ['5m', '15m', '1d'],
            'AMZN': ['5m', '15m', '1d'],
            'PLTR': ['5m', '15m', '1d'],
            'VVV': ['5m', '15m', '1d'],
            }

    indicators = [
            RSI(period=14, magnitude=100),
            MACD(fast=12, slow=26, signal=9),
            OBV(),
            StochasticOscillator(fast_k=14, slow_d=3, slow_k=3)
            ]

    system = System(indicators, fillna=None)
    for stock, intervals in STOCKS.items():
        for interval in intervals:
            directory = os.path.join(OUTPUT_FOLDER, interval)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            STOCK = Stock(stock, interval=interval)
            data = system(STOCK, print_mf=True)
            save_path = os.path.join(directory, f'{stock}.csv')
            data.to_csv(save_path)

    # fig = plt.subplot()
    # fig.plot(data['Close'])

    # buys = data[data['Action'] == 'buy']
    # sells = data[data['Action'] == 'sell']

    # print(f"Number of buys: {len(buys)}")
    # print(f"Number of sells: {len(sells)}")

    # fig.plot(buys['Close'], markersize=5, color='green', marker='o')
    # fig.plot(sells['Close'], markersize=5, color='red', marker='x')
    # fig.set_xticklabels([])
    # plt.show()
    # plt.savefig(f'{STOCK.name}-scuffed.png')

