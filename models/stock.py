from utils import download_history

class Stock:
    def __init__(self, ticker, start=None, end=None, period='1d', interval='1m', **kwargs):
        self.name = ticker
        self.__data = download_history(ticker,
                                       start=start, end=end,
                                       period=period, interval=interval,
                                       **kwargs)

    def __getitem__(self, key):
        return self.__data[key] 

