from utils import download_stock_data

class Stock:
    def __init__(self, ticker, **kwargs):
        self.name = ticker
        self.__data = download_stock_data(ticker, **kwargs)

    def __getitem__(self, key):
        return self.__data[key] 

if __name__ == "__main__":
    STOCK = Stock("AMZN", interval='5m')
    print(STOCK["Close"])
