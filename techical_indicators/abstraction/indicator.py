from models import Stock
from abc import ABC, abstractmethod
from pandas import DataFrame
 
class Indicator(ABC):
    name: str

    @abstractmethod
    def __call__(self, stock: Stock) -> DataFrame:
        raise NotImplementedError
