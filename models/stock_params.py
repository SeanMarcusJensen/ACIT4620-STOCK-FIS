import sys
import json
from .stock import Stock

class StockParams:
    def __init__(self,
                 ticker: str,
                 start: str | None,
                 end: str | None,
                 period: str,
                 interval: str,
                 *args,
                 **kwargs) -> None:
        self.ticker = ticker
        self.start = start
        self.end = end
        self.period = period
        self.interval = interval
        self.args = args
        self.wargs = kwargs

    @staticmethod
    def from_dict(params: dict) -> 'StockParams':
        return StockParams(**params)

    @staticmethod
    def from_command_line() -> 'StockParams':
        try:
            file = sys.argv[1]
            if file.endswith('.json'):
                with open(file) as f:
                    data = json.load(f)
                return StockParams.from_dict(data)

            return StockParams(*sys.argv[1:])
        except Exception:
            print('Usage: python stock_params.py <ticker> <start> <end> <period> <interval> <args> <kwargs>')
            print('Usage: python stock_params.py <file.json>')
            sys.exit(1)

    def get_stock(self) -> Stock:
        return Stock(self.ticker, self.start, self.end, self.period, self.interval, *self.args, **self.wargs)

if __name__ == '__main__':
    params = StockParams.from_command_line()
    stock = params.get_stock()
    print(stock['Close'].head())
