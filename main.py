import numpy as np
from pandas.core.interchange.dataframe_protocol import enum
import skfuzzy as fuzz
import pandas as pd
from typing import List, Dict
from techical_indicators.abstraction import Indicator
from models import Stock
from utils import DateChunker
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
from techical_indicators import RSI, MACD, OBV, StochasticOscillator


class Signal(enum.Enum):
    SELL = 'sell'
    HOLD = 'hold'
    BUY = 'buy'


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
        if signal == Signal.BUY:
            self.buy(price, index)
        elif signal == Signal.SELL:
            self.sell(price, index)

        return self.profits()

    def buy(self, price, index):
        if self.last_action == Signal.SELL or self.last_action is None:
            self.trades += 1
            self.buy_price = price
            self.n_shares = self.value // price
            self.value = self.value % price
            self.value -= self.comission
            self.last_action = Signal.BUY
            self.history.append((index, price, Signal.BUY))

    def sell(self, price, index):
        if self.last_action == Signal.BUY:
            self.trades += 1
            gain = (price * self.n_shares)
            self.biggest_gain = max(self.biggest_gain, gain)
            self.value += gain
            self.value -= self.comission
            self.n_shares = 0
            self.last_action = Signal.SELL
            self.buy_price = 0.0
            self.history.append((index, price, Signal.SELL))

            if price > self.buy_price:
                self.successful_trades += 1

    def profits(self):
        return (self.value + (self.n_shares * self.buy_price)) - self.starting_value

    def __str__(self):
        trades = (self.successful_trades / self.trades) * 100
        return f"Wallet: [{self.value}, @profit={self.profits()}], Trades: [{self.successful_trades}/{self.trades}({trades}%)], Shares: [n:{self.n_shares}, @{self.buy_price}], Last Action: {self.last_action}, Biggest Winner: {self.biggest_gain}"


class Predictor:
    def __init__(self, sets: List[ctrl.Antecedent]):
        self.sets: Dict[str, ctrl.Antecedent] = {fs.label: fs for fs in sets}
        self.simulator: ctrl.ControlSystemSimulation = self.__construct_system()

    def get_signal(self, series: pd.Series) -> Signal:
        self.simulator.inputs({name: series[name] for name in self.sets.keys()})
        self.simulator.compute()
        output = self.simulator.output.get('action', np.nan)
        return self.__get_output_signal(output)

    def __construct_system(self):
        action = ctrl.Consequent(np.arange(0, 101, 1), 'action')
        action['Sell'] = fuzz.trapmf(action.universe, [0, 0, 30, 50])
        action['Hold'] = fuzz.trimf(action.universe, [30, 50, 70])
        action['Buy'] = fuzz.trapmf(action.universe, [50, 70, 100, 100])
        rules = [
                ctrl.Rule(self['macd']['High']   & self['rsi']['Low']    & self['so']['Low']     & self['obv']['High'],   action['Buy']),
                ctrl.Rule(self['macd']['Low']   & self['rsi']['High']    & self['so']['High']     & self['obv']['Low'],  action['Buy']),
                ctrl.Rule(self['macd']['High']  & self['rsi']['Medium']    & self['so']['Medium']     & self['obv']['High'],   action['Buy']),
                ctrl.Rule(self['rsi']['Low']    & self['so']['Low']     & self['obv']['High'],  action['Buy']),
                ctrl.Rule(self['macd']['Low']   & self['rsi']['Medium']    & self['so']['High']  & self['obv']['Low'],   action['Sell']),
                ctrl.Rule(self['rsi']['High']    & self['so']['High']  & self['obv']['Low'],  action['Sell']),
                ctrl.Rule(self['macd']['Low']  & self['rsi']['High']    & self['so']['High'],  action['Sell']),
                ctrl.Rule(self['macd']['Low']   & self['rsi']['Medium'] & self['so']['Medium'],   action['Hold']),
                ctrl.Rule(self['macd']['High']   & self['rsi']['Medium'] & self['so']['Medium']     & self['obv']['Low'],  action['Hold']),
        ]
        system = ctrl.ControlSystem(rules)
        return ctrl.ControlSystemSimulation(system)

    def __get_output_signal(self, value: float) -> Signal:
        if value < 50:
            return Signal.SELL
        elif value > 50:
            return Signal.BUY
        else:
            return Signal.HOLD

    def __getitem__(self, key) -> ctrl.Antecedent:
        return self.sets[key]


class System:
    def __init__(self, indicators: List[Indicator], fillna: float | None = None):
        self.indicators = indicators
        self.predictor = Predictor([i.get_mf() for i in indicators])
        self.fillna = fillna

    def __call__(self, stock: Stock) -> pd.DataFrame:
        import time
        start = time.time()

        wallet = Wallet(10000)
        stock_data = stock.get_data()
        data = pd.concat([stock_data, *[i(stock) for i in self.indicators]], axis=1)

        if self.fillna is not None:
            data.fillna(self.fillna, inplace=True)

        for index, row in data.iterrows():
            action = self.predictor.get_signal(row)
            data.loc[index, 'Action'] = action.value
            returns = wallet.act(action, row['Close'], index)
            data.loc[index, 'Returns'] = returns

        end = time.time()
        delta = end - start
        print(f"System took {delta} seconds to complete")
        print(wallet)

        return data


if __name__ == "__main__":
    STOCK = Stock('PLTR')

    indicators = [
            RSI(period=14, magnitude=100),
            MACD(fast=12, slow=26, signal=9),
            OBV(),
            StochasticOscillator(fast_k=14, slow_d=3, slow_k=3)
            ]

    system = System(indicators, fillna=0)
    data = system(STOCK)
    data.to_csv('PLTR.csv')
