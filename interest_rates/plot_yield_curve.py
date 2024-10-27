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
from interest_rates.short_rate_models import ShortRateModel, VasicekModel
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import psutil


class ZeroCouponBond:
    def __init__(self, T: float):
        self.dt: float = 0.01
        self.a: float = 0.1
        self.b: float = 0.07
        self.r0: float = 0.02
        self.sigma: float = 0.02
        self.M: int = 10_000  # Monte Carlo simulation sample size
        self.T: float = T

    def dX(self) -> float:
        Z: float = np.random.normal()  # Standard normal r.v.
        dX: float = Z * np.sqrt(self.dt)  # X(t) is a Brownian motion
        return dX


    def calc_approx_yield(self) -> float:
        """
        Calculate yield using an approximation of bond price using Monte Carlo simulation
        :param T:
        :return:
        """
        model: ShortRateModel = VasicekModel(
            dt=self.dt,
            a=self.a,
            b=self.b,
            r0=self.r0,
            sigma=self.sigma,
        )
        # Calculate mean as approx of zero-coupon bond price
        Z = sum([np.exp(-model.integrate_r_dt(self.T)) for k in range(self.M)]) / self.M
        # Calculate yield
        y: float = - np.log(Z) / self.T
        print("Zero-coupon bond of maturity T = {} has value Z = {:.3f}".format(self.T, Z))
        return y


    def calc_exact_yield(self) -> float:
        """
        # Calculate yield using exact bond pricing formula for Vasicek model
        :param T:
        :return:
        """
        A = (1 - np.exp(-self.a * self.T)) / self.a
        B = (self.b - 0.5 * self.sigma**2 / self.a**2) * (A - self.T) - self.sigma**2 * A**2 / (4 * self.a)
        Z = np.exp(-A * self.r0 + B)
        y: float = - np.log(Z) / self.T
        return y


if __name__ == '__main__':
    TT = np.arange(1, 10, 1)
    
    tik = time.time()


    bonds = [ZeroCouponBond(T) for T in TT]


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
