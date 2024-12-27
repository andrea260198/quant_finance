from abc import ABC, abstractmethod
from functools import cached_property
from support.quant_dataclass import QuantDataclass


class AbstractOption(QuantDataclass, ABC):
    @cached_property
    def price(self) -> float:
        return self.price_approx(100_000)

    @abstractmethod
    def price_approx(self, N: int) -> float:
        pass