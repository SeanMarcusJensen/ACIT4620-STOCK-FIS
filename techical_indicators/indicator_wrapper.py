import pandas as pd
from models import Stock
from .abstraction import Indicator
from pandas import DataFrame
from typing import List


class IndicatorWrapper(Indicator):
    def __init__(self, indicator: List[Indicator], fillna: float | None = None) -> None:
        self.__indicator = indicator
        self.__fillna = fillna

    def __call__(self, stock: Stock) -> DataFrame:
        data = stock.get_data()

        for indicator in self.__indicator:
            df = indicator(stock)
            df.rename(columns={org: col for org, col in zip(
                df.columns, indicator.column_names)}, inplace=True)
            if self.__fillna is not None:
                df = df.fillna(self.__fillna)
            data = data.join(df)

        return data
