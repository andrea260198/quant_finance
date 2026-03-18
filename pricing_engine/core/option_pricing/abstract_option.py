from abc import ABC, abstractmethod
from enum import Enum, auto, StrEnum
from functools import cached_property
from support.quant_dataclass import ImmutableDataclass


class PricingMethod(StrEnum):
    EXACT = "EXACT"
    BINOMIAL_TREE = "BINOMIAL_TREE"
    MONTE_CARLO = "MONTE_CARLO"
    QUASI_MONTE_CARLO = "QUASI_MONTE_CARLO"


class AbstractOption(ImmutableDataclass, ABC):
    type: str
    pricing_method: PricingMethod = PricingMethod.BINOMIAL_TREE

    @cached_property
    def price(self) -> float:
        match self.pricing_method, self:
            case PricingMethod.EXACT, ExactTrait() as self:
                return self.price_exact()
            case PricingMethod.BINOMIAL_TREE, ApproxTrait() as self:
                return self.price_approx(100_000)
            case PricingMethod.MONTE_CARLO, MonteCarloTrait() as self:
                return self.price_mc_approx(100_000, 0.01)
            case PricingMethod.QUASI_MONTE_CARLO, QuasiMonteCarloTrait() as self:
                raise self.price_qmc_approx(100_000, 0.01)
            case _:
                raise NotImplementedError("Pricing method not defined.")


class ExactTrait:
    @abstractmethod
    def price_exact(self) -> float:
        pass


class ApproxTrait:
    @abstractmethod
    def price_approx(self, N: int) -> float:
        pass


class MonteCarloTrait:
    @abstractmethod
    def price_mc_approx(self, N: int, dt: float) -> float:
        pass


class QuasiMonteCarloTrait:
    @abstractmethod
    def price_qmc_approx(self, N: int, dt: float) -> float:
        pass