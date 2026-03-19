from functools import cached_property
from typing import Literal

from core.contract import Contract, PricingMethod
from core.interest_rates.short_rate_models import ShortRateModel
import numpy as np


class ZeroCouponBond(Contract):
    type: Literal["ZeroCouponBond"]
    T: float  # Maturity
    short_rate_model: ShortRateModel
    M: int = 10_000  # Monte Carlo simulation sample size
    face_value: float = 1.0
    pricing_method: PricingMethod = PricingMethod.QUASI_MONTE_CARLO

    @cached_property
    def price(self) -> float:
        match self.pricing_method:
            case PricingMethod.MONTE_CARLO:
                return self.calc_approx_yield() * self.face_value
            case PricingMethod.QUASI_MONTE_CARLO:
                return self.calc_qmc_approx_yield() * self.face_value
            case PricingMethod.EXACT:
                return self.calc_exact_yield() * self.face_value
            case _:
                raise NotImplementedError("Pricing method not defined.")


    def calc_approx_yield(self) -> float:
        """
        Calculate yield using an approximation of bond price using Monte Carlo simulation
        :param T:
        :return:
        """
        # Calculate mean as approx of zero-coupon bond price
        Z = np.mean([np.exp(-self.short_rate_model.integrate_r_dt(self.T)) for k in range(self.M)])
        # Calculate yield
        y: float = - np.log(Z) / self.T
        print("Zero-coupon bond of maturity T = {} has value Z = {:.3f}".format(self.T, float(Z)))
        return y

    def calc_qmc_approx_yield(self) -> float:
        """
        Calculate yield using an approximation of bond price using Quasi-Monte Carlo simulation
        :return: y
        """
        # Calculate mean as approx of zero-coupon bond price
        Z = np.mean(np.exp(-self.short_rate_model.approximate_integral_with_qmc(self.T, self.M)))
        # Calculate yield
        y: float = - np.log(Z) / self.T
        print("Zero-coupon bond of maturity T = {} has value Z = {:.3f}".format(self.T, float(Z)))
        return y

    def calc_exact_yield(self) -> float:
        return self.short_rate_model.calc_exact_yield(self.T)





