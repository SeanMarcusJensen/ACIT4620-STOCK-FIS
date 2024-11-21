from models import Stock
from abc import ABC, abstractmethod
from typing import Tuple
from pandas import DataFrame

import skfuzzy.control as ctrl

class Indicator(ABC):
    name: str
    column_names: list[str]

    @abstractmethod
    def __call__(self, stock: Stock, fillna: float | None = None) -> Tuple[ctrl.Antecedent, DataFrame]:
        raise NotImplementedError
