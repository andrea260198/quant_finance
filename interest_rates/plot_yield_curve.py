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
from interest_rates.short_rate_models import VasicekModel, ModelParameters
from interest_rates.zero_coupon_bond import ZeroCouponBond
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import psutil


def get_yield_curves(TT) -> None:
    tik = time.time()

    model_parameters = ModelParameters(
        dt=0.01,
        a=0.1,
        b=0.07,
        r0=0.02,
        sigma=0.02,
    )

    M = 100_000
    bonds = [ZeroCouponBond(T=T, short_rate_model=VasicekModel(model_parameters), M=M) for T in TT]

    # Multi-core version
    n_cores = psutil.cpu_count(logical=False)
    pool = Pool(n_cores)
    yy = pool.map(ZeroCouponBond.calc_qmc_approx_yield, bonds)

    # Single-core version
    #yy = list(map(ZeroCouponBond.calc_qmc_approx_yield, bonds))

    tok = time.time()

    print('Time = ', tok - tik, 's')

    yy2 = list(map(ZeroCouponBond.calc_exact_yield, bonds))

    return yy, yy2

def plot_yield_curves(TT, yy, yy2) -> None:
    plt.plot(TT, yy)
    plt.plot(TT, yy2, 'k:')
    plt.ylim(0, 0.04)
    plt.xlabel("Maturity [y]")
    plt.ylabel("Yield")
    plt.legend(["Quasi-Monte Carlo", "closed-form"])
    plt.show()


if __name__ == '__main__':
    TT = np.arange(1, 10, 1)

    yy, yy2 = get_yield_curves(TT)

    plot_yield_curves(TT, yy, yy2)
