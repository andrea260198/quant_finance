from abc import ABC, abstractmethod
from enum import Enum, auto, StrEnum
from functools import cached_property
from support.quant_dataclass import QuantDataclass


class PricingMethod(StrEnum):
    EXACT = "EXACT"
    BINOMIAL_TREE = "BINOMIAL_TREE"
    MONTE_CARLO = "MONTE_CARLO"
    QUASI_MONTE_CARLO = "QUASI_MONTE_CARLO"


class AbstractOption(QuantDataclass, ABC):
    type: str
    pricing_method: PricingMethod = PricingMethod.BINOMIAL_TREE

    @cached_property
    def price(self) -> float:
        match self.pricing_method:
            case PricingMethod.EXACT:
                return self.price_exact()
            case PricingMethod.BINOMIAL_TREE:
                return self.price_approx(100_000)
            case PricingMethod.MONTE_CARLO:
                return self.price_mc_approx(100_000, 0.01)
            case PricingMethod.QUASI_MONTE_CARLO:
                raise NotImplementedError()
            case _:
                raise ValueError("Pricing method not defined.")

    @abstractmethod
    def price_approx(self, N: int) -> float:
        pass