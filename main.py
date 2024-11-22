import numpy as np
from pandas.core.interchange.dataframe_protocol import enum
import skfuzzy as fuzz
import pandas as pd
from typing import List, Dict
from techical_indicators.abstraction import Indicator
from models import Stock
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
from techical_indicators import RSI, MACD, OBV, StochasticOscillator
from dataclasses import dataclass


class Signal(enum.Enum):
    STRONGSELL = 'strong_sell'
    SELL = 'sell'
    HOLD = 'hold'
    BUY = 'buy'
    STRONGBUY = 'strong_buy'


@dataclass
class Transaction:
    time: pd.Timestamp
    shares: int
    at_price: float
    action: Signal

    @staticmethod
    def create_sell(time, shares, price) -> tuple:
        transaction = Transaction(time, -shares, price, Signal.SELL)
        return (transaction, -1 * transaction.cost())

    @staticmethod
    def create_buy(time, shares, price) -> tuple:
        transaction = Transaction(time, shares, price, Signal.BUY)
        return (transaction, transaction.cost())

    def cost(self):
        return self.shares * self.at_price


class Wallet:
    def __init__(self, capital, ticker: str):
        self.ticker = ticker
        self.deposit = capital
        self.capital = capital
        self.comission = 0.0
        self.biggest_gain = 0.0
        self.transactions: List[Transaction] = []
        self.n_shares = 0

    def act(self, signal: Signal, price, index) -> tuple:
        match signal:
            case Signal.STRONGBUY:
                return self.buy(price, index, .4)
            case Signal.BUY:
                return self.buy(price, index, .1)
            case Signal.SELL:
                return self.sell(price, index, .1)
            case Signal.STRONGSELL:
                return self.sell(price, index, .4)
        return 0.0, 0
    
    def strongbuy(self, price, index, pct = 0.4):
        buy_n_shares = (self.capital * pct) // price
        if buy_n_shares < 1:
            return 0.0, 0

        transaction, cost = Transaction.create_buy(index, buy_n_shares, price)
        self.transactions.append(transaction)

        change = (-1 * cost) / self.capital
        self.capital -= cost
        self.capital -= self.comission
        self.n_shares += buy_n_shares
        return change, buy_n_shares


    def buy(self, price, index, pct=1.0) -> tuple:
        buy_n_shares = (self.capital * pct) // price
        if buy_n_shares < 1:
            return 0.0, 0

        transaction, cost = Transaction.create_buy(index, buy_n_shares, price)
        self.transactions.append(transaction)

        change = (-1 * cost) / self.capital
        self.capital -= cost
        self.capital -= self.comission
        self.n_shares += buy_n_shares
        return change, buy_n_shares

    def sell(self, price, index, pct=1.0) -> tuple:
        import math
        sell_n_shares = math.floor(self.n_shares * pct)
        if sell_n_shares < 1:
            return 0.0, 0

        transaction, gain = Transaction.create_sell(
            index, sell_n_shares, price)
        self.transactions.append(transaction)

        change = gain / self.capital
        self.biggest_gain = max(self.biggest_gain, gain)
        self.capital += gain
        self.capital -= self.comission
        self.n_shares -= sell_n_shares
        return change, (-1 * sell_n_shares)

    def current_value(self, price):
        total_shares = sum([t.shares for t in self.transactions])
        return (total_shares * price) + self.capital

    def accumulate_cost(self):
        return sum([t.cost() for t in self.transactions])

    def profits(self, price=None):
        if price is None:
            if len(self.transactions) == 0:
                return 0
            price = self.transactions[-1].at_price
        return self.current_value(price) - self.deposit

    def get_transactions(self):
        return self.transactions

    def get_info(self, current_market_value: float) -> dict:
        return {
            'ticker': self.ticker,
            'capital': self.capital,
            'deposit': self.deposit,
            'current_market_value': self.current_value(current_market_value),
            'profits': self.profits(current_market_value),
            'number_of_shares': self.n_shares,
            'number_of_trades': len(self.transactions),
            'yield_pct': self.profits(current_market_value) / self.deposit,
        }

    def __str__(self):
        sell_trades = sum(
            [1 for t in self.transactions if t.action == Signal.SELL])
        buy_trades = sum(
            [1 for t in self.transactions if t.action == Signal.BUY])
        n_trades = sell_trades + buy_trades
        return f"Wallet: [capital: {self.capital}, starting capital: {self.deposit}], Trades: [{n_trades} @ s:{sell_trades}, b:{buy_trades}]" \
            f"Shares: [{self.n_shares}], Biggest Winner: [{self.biggest_gain}]"


class Predictor:
    def __init__(self, sets: List[ctrl.Antecedent]):
        self.sets: Dict[str, ctrl.Antecedent] = {fs.label: fs for fs in sets}
        self.action = ctrl.Consequent(np.arange(0, 31, 1), 'action')
        self.action['Strongsell'] = fuzz.trimf(self.action.universe, [0, 3, 6])
        self.action['Sell'] = fuzz.trimf(self.action.universe, [6, 9, 12])
        self.action['Hold'] = fuzz.trimf(self.action.universe, [12, 15, 18])
        self.action['Buy'] = fuzz.trimf(self.action.universe, [18, 21, 24])
        self.action['Strongbuy'] = fuzz.trimf(self.action.universe, [24, 27, 30])

        self.simulator: ctrl.ControlSystemSimulation = self.__construct_system()

    def get_signal(self, series: pd.Series, plot_signal: bool = False) -> Signal:
        self.simulator.inputs({name: series[name]
                              for name in self.sets.keys()})
        self.simulator.compute()
        output = self.simulator.output.get('action', np.nan)
        if plot_signal:
            self.simulator.compute()
            self.action.view(sim=self.simulator)
            plt.show()
        return self.__get_output_signal(output)

    def __construct_system(self):

        rules = [
            ctrl.Rule(self['macd']['High'] & self['rsi']['Low'] & self['so']
                      ['Low'] & self['obv']['High'],   self.action['Buy']),
            ctrl.Rule(self['macd']['Low'] & self['rsi']['High'] & self['so']
                      ['High'] & self['obv']['Low'],  self.action['Buy']),
            ctrl.Rule(self['macd']['High'] & self['rsi']['Medium'] & self['so']
                      ['Medium'] & self['obv']['High'],   self.action['Buy']),
            ctrl.Rule(self['rsi']['Low'] & self['so']['Low'] &
                      self['obv']['High'],  self.action['Buy']),
            ctrl.Rule(self['macd']['Low'] & self['rsi']['Medium'] & self['so']
                      ['High'] & self['obv']['Low'],   self.action['Sell']),
            ctrl.Rule(self['rsi']['High'] & self['so']['High'] &
                      self['obv']['Low'],  self.action['Sell']),
            ctrl.Rule(self['macd']['Low'] & self['rsi']['High']
                      & self['so']['High'],  self.action['Sell']),
            ctrl.Rule(self['macd']['Low'] & self['rsi']['Medium']
                      & self['so']['Medium'],   self.action['Hold']),
            ctrl.Rule(self['macd']['High'] & self['rsi']['Medium'] & self['so']
                      ['Medium'] & self['obv']['Low'],  self.action['Hold']),
        ]

        system = ctrl.ControlSystem(rules)

        return ctrl.ControlSystemSimulation(system)

    def __get_output_signal(self, value: float) -> Signal:
        if value <= 8:
            return Signal.STRONGSELL
        elif value <= 11:
            return Signal.SELL
        elif value <= 19:
            return Signal.BUY
        elif value <= 22:
            return Signal.STRONGBUY
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

        wallet = Wallet(10000, stock.name)
        mfs, indicators = zip(*[i(stock, self.fillna)
                              for i in self.indicators])
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

            wallet_change_pct, shares = wallet.act(action, row['Close'], index)
            data.loc[index, 'change_pct'] = wallet_change_pct
            data.loc[index, 'change_shares'] = shares
            data.loc[index, 'profits'] = wallet.profits(row['Close'])
            data.loc[index, 'shares'] = wallet.n_shares
            data.loc[index, 'curr_capital'] = wallet.capital

        end = time.time()
        delta = end - start
        print(f"System took {delta} seconds to complete")
        print(wallet.get_info(data['Close'].iloc[-1]))
        print(data.head())

        return data


if __name__ == "__main__":
    import os
    OUTPUT_FOLDER = 'output'
    STOCKS = {
        'AAPL': [('1d', '5m'), ('1d', '15m'), ('1y', '1d')],
        'AMZN': [('1d', '5m'), ('1d', '15m'), ('1y', '1d')],
        'PLTR': [('1d', '5m'), ('1d', '15m'), ('1y', '1d')],
        'VVV': [('1d', '5m'), ('1d', '15m'), ('1y', '1d')],
    }

    # STOCKS = {
    #         'AAPL': [('1y', '1d')],
    #         'AMZN': [('1y', '1d')],
    #         'PLTR': [('1y', '1d')],
    #         'VVV': [('1y',' 1d')],
    #         }

    indicators = [
        RSI(period=14, magnitude=100),
        MACD(fast=12, slow=26, signal=9),
        OBV(),
        StochasticOscillator(fast_k=14, slow_d=3, slow_k=3)
    ]

    system = System(indicators, fillna=None)
    for stock, intervals in STOCKS.items():
        for period, interval in intervals:
            directory = os.path.join(OUTPUT_FOLDER, interval)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            STOCK = Stock(stock, interval=interval, period=period)
            data = system(STOCK, print_mf=False)
            save_path = os.path.join(directory, f'{stock}.csv')
            data.to_csv(save_path)
