from functools import cached_property
from typing import Literal, Union, Annotated

from pydantic import Field
from pydantic_extra_types.currency_code import ISO4217

from core.contract import Contract, PricingMethod
from core.interest_rates.short_rate_models import VasicekModel, CoxIngersolRossModel
import numpy as np


class ZeroCouponBond(Contract):
    type: Literal["ZeroCouponBond"]
    T: float  # Maturity
    short_rate_model: Annotated[Union[VasicekModel, CoxIngersolRossModel], Field(discriminator="type")]
    pricing_method: PricingMethod
    M: int = -1 # Monte Carlo simulation sample size
    face_value: float = 1.0
    currency: ISO4217 = "USD"
    issuer: str = "GenericIssuer"

    @cached_property
    def price(self) -> float:
        match self.pricing_method:
            case PricingMethod.MONTE_CARLO:
                return self.price_approx() * self.face_value
            case PricingMethod.QUASI_MONTE_CARLO:
                return self.price_qmc_approx() * self.face_value
            case PricingMethod.EXACT:
                return self.price_exact() * self.face_value
            case _:
                raise NotImplementedError("Pricing method not defined.")

    @cached_property
    def bond_yield(self) -> float:
        return - np.log(self.price / self.face_value) / self.T

    def get_yield(self) -> float:
        return self.bond_yield

    def price_approx(self) -> float:
        """
        Calculate yield using an approximation of bond price using Monte Carlo simulation
        :param T:
        :return:
        """
        # Calculate mean as approx of zero-coupon bond price
        Z = np.mean([np.exp(-self.short_rate_model.integrate_r_dt(self.T)) for k in range(self.M)])
        return Z.item()

    def price_qmc_approx(self) -> float:
        """
        Calculate yield using an approximation of bond price using Quasi-Monte Carlo simulation
        :return: y
        """
        # Calculate mean as approx of zero-coupon bond price
        Z = np.mean(np.exp(-self.short_rate_model.approximate_integral_with_qmc(self.T, self.M)))
        return Z.item()

    def price_exact(self) -> float:
        return np.exp(- self.short_rate_model.calc_exact_yield(self.T) * self.T) * self.face_value





