from abc import ABC, abstractmethod
from enum import Enum, auto
from functools import cached_property
from support.quant_dataclass import QuantDataclass


class PricingMethod(Enum):
    EXACT = auto()
    APPROX = auto()


class AbstractOption(QuantDataclass, ABC):
    pricing_method: PricingMethod = PricingMethod.APPROX

    @cached_property
    def price(self) -> float:
        match self.pricing_method:
            case PricingMethod.EXACT:
                return self.price_exact()
            case PricingMethod.APPROX:
                return self.price_approx(100_000)
            case _:
                raise ValueError("Pricing method not defined.")

    @abstractmethod
    def price_approx(self, N: int) -> float:
        pass