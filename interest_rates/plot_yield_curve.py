#!/usr/bin/python3
#
# Author: Andrea Cassotti
# Version: 1.0
# Python: 3.8
#
# Summary:
# The following code plots a yield curve obtained from the CIR model
# for instantaneus risk-free rate.
# Below is the CIR model stochastic differential equation:
# dr = a(b - r)dt + sigma sqrt(r) dX
# 
# The zero-coupon bond price is obtained using Monte-Carlo simulation as 
# follows:
# B() = E[exp(-int(r(t)dt,0,T))]
#
# The yield is obtained as follows:
# B = exp(-y T)  =>  y = - log(B) / T
#

import time
from interest_rates.short_rate_models import ShortRateModel, VasicekModel, ModelParameters
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import psutil


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


if __name__ == '__main__':
    TT = np.arange(1, 10, 1)
    
    tik = time.time()


    model_parameters = ModelParameters(
        dt=0.01,
        a=0.1,
        b=0.07,
        r0=0.02,
        sigma=0.02,
    )

    bonds = [ZeroCouponBond(T, VasicekModel(model_parameters)) for T in TT]


    # Multi-core version
    n_cores = psutil.cpu_count(logical=False)
    pool = Pool(n_cores)
    yy = pool.map(ZeroCouponBond.calc_approx_yield, bonds)
    
    # Single-core version
    #yy = list(map(ZeroCouponBond.calc_approx_yield, bonds))
    
    tok = time.time()
    
    print('Time = ', tok-tik, 's')
    
    yy2 = list(map(ZeroCouponBond.calc_exact_yield, bonds))

    plt.plot(TT, yy)
    plt.plot(TT, yy2, 'k:')
    plt.ylim(0, 0.04)
    plt.xlabel("Maturity [y]")
    plt.ylabel("Yield")
    plt.legend(["Monte Carlo", "closed-form"])
    plt.show()
