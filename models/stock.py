import pandas as pd
from utils import download_stock_data
from utils import DateChunker

class Stock:
    def __init__(self, ticker, **kwargs):
        self.name = ticker
        self.__data = download_stock_data(ticker, **kwargs)
        self.__chunker = DateChunker(self.get_data(), index='Datetime', column='Date')

    def by_date(self) -> list:
        return self.__chunker

    def get_on_date(self, date: str) -> pd.DataFrame:
        return self.__chunker.get_date(date)

    def get_data(self) -> pd.DataFrame:
        return self.__data

    def __getitem__(self, key):
        return self.__data[key]

    @property
    def index(self):
        return self.__data.index


if __name__ == "__main__":
    STOCK = Stock("AAPL", interval='5m')
    print(STOCK.get_data().info())

    for chunk in STOCK.by_date():
        print(chunk.head())

    print(STOCK.get_on_date('2024-11-01').head())
