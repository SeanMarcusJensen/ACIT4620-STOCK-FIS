import pandas as pd
from typing import Dict

class DateChunker:
    def __init__(self, data: pd.DataFrame, column: str = 'Date') -> None:
        self.chunked: Dict[str, pd.DataFrame] = {}
        self.columns = data.columns

        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Datetime'], utc=True).dt.date
        data.set_index('Datetime', inplace=True)
        chunks = data.groupby(column)
        for key, value in chunks:
            self.chunked[key.__str__()] = value

    def get_date(self, date: str) -> pd.DataFrame:
        data = self.chunked.get(date, pd.DataFrame())
        data = data[self.columns]
        return pd.DataFrame(data)
    
    def __iter__(self):
        return iter(self.chunked.values())

if __name__ == "__main__":
    from models import Stock
    STOCK = Stock("AAPL", interval='5m')
    chunker = DateChunker(STOCK.get_data())
    chunk = chunker.get_date('2024-11-01')
    print(chunk.head())

