#!/usr/bin/python3
#
# Author: Andrea Cassotti
# Version: 1.0
# Python: 3.14
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

from core.contract import PricingMethod
from core.interest_rates.short_rate_models import VasicekModel, ShortRateModel
from core.interest_rates.zero_coupon_bond import ZeroCouponBond
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import psutil


def main():
    TT = np.arange(1., 10., 1.).tolist()

    yy, yy2 = get_approx_and_exact_yield_curves(TT)

    plot_yield_curves(TT, yy, yy2)


def get_yield_curve(TT: list[float], short_rate_model: ShortRateModel, pricing_method: PricingMethod, M=-1) -> list[float]:
    bonds = [ZeroCouponBond(type="ZeroCouponBond", T=T, short_rate_model=short_rate_model, M=M, pricing_method=pricing_method) for T in TT]

    # Multi-core version
    n_cores = psutil.cpu_count(logical=False)
    pool = Pool(n_cores)
    yy = pool.map(ZeroCouponBond.get_yield, bonds)

    # Single-core version
    #yy = list(map(ZeroCouponBond.get_yield, bonds))

    return [y.item() for y in yy]


def get_approx_and_exact_yield_curves(TT: list[float]) -> tuple[list[float], list[float]]:
    tik = time.time()

    M = 100_000
    bonds = [
        ZeroCouponBond(
            type="ZeroCouponBond",
            T=T,
            short_rate_model=VasicekModel(
                type="VasicekModel",
                dt=0.01,
                a=0.1,
                b=0.07,
                r0=0.02,
                sigma=0.02,
            ),
            M=M,
            pricing_method=PricingMethod.QUASI_MONTE_CARLO
        ) for T in TT
    ]

    exact_bonds = [
        ZeroCouponBond(
            type="ZeroCouponBond",
            T=T,
            short_rate_model=VasicekModel(
                type="VasicekModel",
                dt=0.01,
                a=0.1,
                b=0.07,
                r0=0.02,
                sigma=0.02,
            ),
            pricing_method=PricingMethod.EXACT
        ) for T in TT
    ]

    # Multi-core version
    n_cores = psutil.cpu_count(logical=False)
    pool = Pool(n_cores)
    yy_approx = pool.map(ZeroCouponBond.get_yield, bonds)

    # Single-core version
    #yy_approx = list(map(ZeroCouponBond.calc_qmc_approx_yield, bonds))

    tok = time.time()

    print('Time = ', tok - tik, 's')

    yy_exact = list(map(ZeroCouponBond.get_yield, exact_bonds))

    return yy_approx, yy_exact

def plot_yield_curves(TT: list[float], yy: list[float], yy2: list[float]) -> None:
    plt.plot(TT, yy)
    plt.plot(TT, yy2, 'k:')
    plt.ylim(0, 0.04)
    plt.xlabel("Maturity [y]")
    plt.ylabel("Yield")
    plt.legend(["Quasi-Monte Carlo", "closed-form"])
    plt.show()


if __name__ == '__main__':
    main()
