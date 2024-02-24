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
from interest_rates.int_rate_models import ShortRateModel, VasicekModel
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt


dt = 0.01
a = 0.1
b = 0.07
r0 = 0.02
sigma = 0.02
M = 20_000  # Monte Carlo simulation sample size


def dX() -> float:
    Z: float = np.random.normal()  # Standard normal r.v.
    dX: float = Z * np.sqrt(dt)  # X(t) is a Brownian motion
    return dX


def calc_approx_yield(T: float) -> float:
    """
    Calculate yield using an approximation of bond price using Monte Carlo simulation
    :param T:
    :return:
    """
    model: ShortRateModel = VasicekModel(
        dt=0.01,
        a=0.1,
        b=0.07,
        r0=0.02,
        sigma=0.02,
    )
    # Calculate mean as approx of zero-coupon bond price
    B = sum([np.exp(-model.integrate_r_dt(T)) for k in range(M)]) / M
    # Calculate yield
    y: float = - np.log(B) / T
    print("Yield calculated for T = ", T, " : E[exp(-int(rdt))] = ", B)
    return y
    

def calc_exact_yield(T: float) -> float:
    """
    # Calculate yield using exact bond pricing formula for Vasicek model
    :param T:
    :return:
    """
    A = (1 - np.exp(-a * T)) / a
    B = (b - 0.5 * sigma**2 / a**2) * (A - T) - sigma**2 * A**2 / (4 * a)
    Z = np.exp(-A * r0 + B)
    y: float = - np.log(Z) / T
    return y


if __name__ == '__main__':
    TT = np.arange(1, 10, 1)
    
    tik = time.time()

    # Multi-core version    
    pool = Pool(6)
    yy = pool.map(calc_approx_yield, TT)
    
    # Single-core version
    #yy = np.array(list(map(bond_yield, TT)))
    
    tok = time.time()
    
    print('Time = ', tok-tik, 's')
    
    yy2 = list(map(calc_exact_yield, TT))

    plt.plot(TT, yy)
    plt.plot(TT, yy2, 'k:')
    plt.ylim(0, 0.04)
    plt.show()
