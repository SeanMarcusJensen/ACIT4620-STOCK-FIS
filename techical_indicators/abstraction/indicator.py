from models import Stock
from abc import ABC, abstractmethod
from pandas import DataFrame

import skfuzzy.control as ctrl

class Indicator(ABC):
    name: str
    column_names: list[str]

    @abstractmethod
    def __call__(self, stock: Stock) -> DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_mf(self) -> ctrl.Antecedent:
        raise NotImplementedError()
