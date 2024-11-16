from utils import download_stock_data
import pandas as pd


class Stock:
    def __init__(self, ticker, **kwargs):
        self.name = ticker
        self.__data = download_stock_data(ticker, **kwargs)
        print(self.__data.iloc[-1])

    def get_date(self, date: str) -> pd.DataFrame:
        return self.__data.query(f'Datetime >= "{date}"')

    def get_data(self) -> pd.DataFrame:
        return self.__data

    def __getitem__(self, key):
        return self.__data[key]

    @property
    def index(self):
        return self.__data.index


if __name__ == "__main__":
    STOCK = Stock("AMZN", interval='5m')
    print(STOCK["Close"])
