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
                self.buy(price, index, .2)
            case Signal.SELL:
                self.sell(price, index, .2)
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
        action = ctrl.Consequent(np.arange(0, 31, 1), 'action')
        action['Sell'] = fuzz.trimf(action.universe, [0, 5, 10])
        action['Hold'] = fuzz.trimf(action.universe, [10, 15, 20])
        action['Buy'] = fuzz.trimf(action.universe, [20, 25, 30])

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

    def __call__(self, stock: Stock) -> pd.DataFrame:
        import time
        start = time.time()

        wallet = Wallet(10000)
        mfs, indicators = zip(*[i(stock) for i in self.indicators])
        data = pd.concat([stock.get_data(), *[i for i in indicators]], axis=1)

        predictor = Predictor([mf for mf in mfs])

        if self.fillna is not None:
            data.fillna(self.fillna, inplace=True)
        else:
            data.dropna(inplace=True)

        for index, row in data.iterrows():
            action = predictor.get_signal(row)
            data.loc[index, 'Action'] = action.value
            returns = wallet.act(action, row['Close'], index)
            data.loc[index, 'Returns'] = returns

        end = time.time()
        delta = end - start
        print(f"System took {delta} seconds to complete")
        print(wallet)
        print(data.head())

        return data


if __name__ == "__main__":
    STOCK = Stock('VVV')

    indicators = [
            RSI(period=14, magnitude=100),
            MACD(fast=12, slow=26, signal=9),
            OBV(),
            StochasticOscillator(fast_k=14, slow_d=3, slow_k=3)
            ]

    system = System(indicators, fillna=None)
    data = system(STOCK)
    data.to_csv(f'{STOCK.name}.csv')

    fig = plt.subplot()
    fig.plot(data['Close'])

    buys = data[data['Action'] == 'buy']
    sells = data[data['Action'] == 'sell']

    print(f"Number of buys: {len(buys)}")
    print(f"Number of sells: {len(sells)}")

    fig.plot(buys['Close'], markersize=5, color='green', marker='o')
    fig.plot(sells['Close'], markersize=5, color='red', marker='x')
    fig.set_xticklabels([])
    plt.show()
    plt.savefig(f'{STOCK.name}-scuffed.png')

