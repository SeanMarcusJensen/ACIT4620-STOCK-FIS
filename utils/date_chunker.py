import pandas as pd
from typing import Dict


class DateChunker:
    def __init__(self, data: pd.DataFrame, index: str = 'Datetime', column: str = 'Date') -> None:
        self.chunked: Dict[str, pd.DataFrame] = {}
        self.columns = data.columns

        data = data.reset_index()
        data[column] = pd.to_datetime(data[index], utc=True).dt.date
        data.set_index(index, inplace=True)
        chunks = data.groupby(column)
        for key, value in chunks:
            self.chunked[key.__str__()] = pd.DataFrame(value[self.columns])

    def get_date(self, date: str) -> pd.DataFrame:
        data = self.chunked.get(date, pd.DataFrame())
        return data

    def get_date_chunk(self, start: str, end: str) -> pd.DataFrame:
        data = pd.concat([self.get_date(date)
                         for date in self.chunked.keys() if start <= date <= end])
        return data

    def __iter__(self):
        return iter(self.chunked.values())
    
    def get_data(self) -> pd.DataFrame:
        return pd.concat([chunk for chunk in self.chunked.values()])


if __name__ == "__main__":
    from models import Stock
    STOCK = Stock("AAPL", interval='5m')
    chunker = DateChunker(STOCK.get_data())
    chunk = chunker.get_date('2024-11-01')
    print(chunk.head())
