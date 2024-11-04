from utils import download_history
import matplotlib.pyplot as plt

class Stock:
    def __init__(self, ticker, start=None, end=None, period='1d', interval='1m'):
        self.ticker = ticker
        self.data = download_history(ticker,
                                     start=start, end=end,
                                     period=period, interval=interval)

    def plot(self, key):
        plt.plot(self[key])
        plt.show()

    def __getitem__(self, key):
        return self.data[key] 


if __name__ == "__main__":
    AAPL = Stock('AAPL')
    AAPL.plot('Close')
