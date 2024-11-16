from models import Stock
from abc import ABC, abstractmethod
from pandas import DataFrame


class Indicator(ABC):
    name: str
    column_names: list[str]

    @abstractmethod
    def __call__(self, stock: Stock) -> DataFrame:
        raise NotImplementedError
