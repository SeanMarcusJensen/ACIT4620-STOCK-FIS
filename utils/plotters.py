import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def plot_buy_sell_signals(data, title: str):
    plt.figure(figsize=(18,4))
    plt.plot(data['Close'], color='lightgray', label='Close price', zorder=1)
    sell = data.loc[data['action'] == 'sell']
    buy = data.loc[data['action'] == 'buy']

    sells = len(sell)
    buys = len(buy)
    holds = len(data) - sells - buys

    print(f'Total buys: {buys}')
    print(f'Total holds: {holds}')
    print(f'Total sales: {sells}')

    plt.title(title)
    plt.scatter(buy.index, buy['Close'], color='g', label='Buy', marker='^', zorder=2)
    plt.scatter(sell.index, sell['Close'], color='r', label='Sell', marker='v', zorder=2)
    plt.legend()
    plt.show()

def plot_buy_sell_actions(data, title: str):
    plt.figure(figsize=(18,4))
    plt.plot(data['Close'], color='lightgray', label='Close price', zorder=1)
    sell = data.loc[(data['action'] == 'sell') & (data['change_pct'] != 0)]
    buy = data.loc[(data['action'] == 'buy') & (data['change_pct'] != 0)]

    sells = len(sell)
    buys = len(buy)
    holds = len(data) - sells - buys

    print(f'Total buys: {buys}')
    print(f'Total holds: {holds}')
    print(f'Total sales: {sells}')

    plt.title(title)
    plt.scatter(buy.index, buy['Close'], color='g', label='Buy', marker='^', zorder=2)
    plt.scatter(sell.index, sell['Close'], color='r', label='Sell', marker='v', zorder=2)
    plt.legend()
    plt.show()

def plot_profits(data, title: str):
    plt.plot(data['profits'], color='orange', label='Profits over time')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_indicators(data, title: str):
    indicators = ['rsi', 'macd', 'obv', 'so']
    data_normalization = data[indicators]

    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(data_normalization)

    normalized_df = pd.DataFrame(normalized, columns=indicators)

    plt.figure(figsize=(22,2))
    for column in normalized_df.columns:
        plt.plot(data['Datetime'], normalized_df[column], label=column)

    plt.xlabel("Time")
    plt.ylabel("Normalized Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.tick_params(labelbottom=False)
    plt.title(title)
    plt.show()