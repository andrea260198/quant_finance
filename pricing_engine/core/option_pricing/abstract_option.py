from abc import ABC, abstractmethod
from functools import cached_property
from core.contract import Contract, PricingMethod
from pydantic_extra_types.currency_code import ISO4217


class AbstractOption(Contract, ABC):
    T: float
    r: float
    S_0: float
    sigma: float
    pricing_method: PricingMethod
    underlying_name: str = "GenericUnderlying"
    underlying_quantity: float = 1
    currency: ISO4217 = "USD"

    @cached_property
    def price(self) -> float:
        match self.pricing_method, self:
            case PricingMethod.EXACT, ExactTrait() as self:
                return self.price_exact() * self.underlying_quantity
            case PricingMethod.BINOMIAL_TREE, ApproxTrait() as self:
                return self.price_approx(100_000) * self.underlying_quantity
            case PricingMethod.MONTE_CARLO, MonteCarloTrait() as self:
                return self.price_mc_approx(100_000, 0.01) * self.underlying_quantity
            case PricingMethod.QUASI_MONTE_CARLO, QuasiMonteCarloTrait() as self:
                return self.price_qmc_approx(100_000, 0.01) * self.underlying_quantity
            case _:
                raise NotImplementedError("Pricing method not defined.")

    @cached_property
    def delta(self) -> float:
        return self._compute_second_order_difference(self, "S_0")

    @cached_property
    def theta(self) -> float:
        return - self._compute_second_order_difference(self, "T")

    @cached_property
    def vega(self) -> float:
        return self._compute_second_order_difference(self, "sigma")

    @cached_property
    def rho(self) -> float:
        return self._compute_second_order_difference(self, "r")

    @cached_property
    def gamma(self) -> float:
        return self._compute_second_derivative(self, "S_0")

    @staticmethod
    def _compute_second_order_difference(self, option_param: str) -> float:
        # Compute second-order central difference approximation
        EPSILON = 0.01
        option_0 = self.model_copy(update={option_param: getattr(self, option_param) - EPSILON})
        option_1 = self.model_copy(update={option_param: getattr(self, option_param) + EPSILON})
        return (option_1.price - option_0.price) / (2 * EPSILON)

    @staticmethod
    def _compute_second_derivative(self, option_param: str) -> float:
        EPSILON = 0.01
        option_0 = self.model_copy(update={option_param: getattr(self, option_param) - EPSILON})
        option_1 = self.model_copy(update={option_param: getattr(self, option_param) + EPSILON})
        return (option_1.price - 2 * self.price + option_0.price) / (EPSILON ** 2)


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
