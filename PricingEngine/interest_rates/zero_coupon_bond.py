from interest_rates.short_rate_models import ShortRateModel
import numpy as np
from support.quant_dataclass import ImmutableDataclass


class ZeroCouponBond(ImmutableDataclass):
    T: float
    short_rate_model: ShortRateModel
    M: int = 10_000  # Monte Carlo simulation sample size

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
        print("Zero-coupon bond of maturity T = {} has value Z = {:.3f}".format(self.T, Z))
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
        print("Zero-coupon bond of maturity T = {} has value Z = {:.3f}".format(self.T, Z))
        return y

    def calc_exact_yield(self) -> float:
        return self.short_rate_model.calc_exact_yield(self.T)





