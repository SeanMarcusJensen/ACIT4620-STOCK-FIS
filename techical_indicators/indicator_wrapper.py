import pandas as pd
from models import Stock
from .abstraction import Indicator
from pandas import DataFrame
from typing import List

class IndicatorWrapper(Indicator):
    def __init__(self, indicator: List[Indicator]) -> None:
        self.__indicator = indicator

    def __call__(self, stock: Stock) -> DataFrame:
        indicators = {}

        for indicator in self.__indicator:
            indicators[indicator.name] = indicator(stock)

        return pd.DataFrame(indicators)
