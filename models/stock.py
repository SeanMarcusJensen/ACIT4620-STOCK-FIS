import pandas as pd
from utils import download_stock_data
from utils import DateChunker


class Stock:
    def __init__(self, ticker, **kwargs):
        self.name = ticker
        self.__data = download_stock_data(ticker, **kwargs)
        self.__data = self.__data.drop(columns=['Dividends', 'Stock Splits'])
        # self.__chunker = DateChunker(
        #     self.get_data(), index='Datetime', column='Date')

    def get_data(self) -> pd.DataFrame:
        return self.__data

    def __getitem__(self, key):
        return self.__data[key]

    @property
    def index(self):
        return self.__data.index
