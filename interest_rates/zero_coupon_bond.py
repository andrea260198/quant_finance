import numpy as np
from interest_rates.short_rate_models import ShortRateModel


class ZeroCouponBond:
    def __init__(self, T: float, short_rate_model: ShortRateModel):
        self.M: int = 10_000  # Monte Carlo simulation sample size
        self.T: float = T
        self.short_rate_model: ShortRateModel = short_rate_model

    def calc_approx_yield(self) -> float:
        """
        Calculate yield using an approximation of bond price using Monte Carlo simulation
        :param T:
        :return:
        """
        model: ShortRateModel = self.short_rate_model
        # Calculate mean as approx of zero-coupon bond price
        Z = sum([np.exp(-model.integrate_r_dt(self.T)) for k in range(self.M)]) / self.M
        # Calculate yield
        y: float = - np.log(Z) / self.T
        print("Zero-coupon bond of maturity T = {} has value Z = {:.3f}".format(self.T, Z))
        return y

    def calc_exact_yield(self) -> float:
        return self.short_rate_model.calc_exact_yield(self.T)