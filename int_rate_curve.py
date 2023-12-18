#!/usr/bin/python3
# Author:
# Version:
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
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt


dt = 0.01
a = 0.1
b = 0.07
r0 = 0.02
sigma = 0.02
M = 5000  # Monte Carlo simulation sample size

Z = lambda: np.random.normal()  # Standard normal r.v.
dX = lambda: Z() * np.sqrt(dt)  # X(t) is a Brownian motion


def integrate_r_dt(T: float) -> float:    
    N = int(T / dt)

    r = np.zeros(N)
    
    r[0] = r0
    
    for k in range(N - 1):
        # CIR model
        #dr = a * (b - r[k]) * dt + sigma * np.sqrt(r[k]) * dX()
        # Vasicek model
        dr = a * (b - r[k]) * dt + sigma * dX()
        r[k+1] = r[k]+dr
        
    integral = sum(r * dt)
    return integral


def bond_yield(T: float) -> float:
    B = lambda: np.exp(-integrate_r_dt(T))

    BB = [B() for k in range(M)]

    E = sum(BB) / M

    y = - np.log(E) / T
    
    print("Yield calculated for T = ", T, " E[exp(-int(rdt))] = ", E)
    
    return y
    
    

def exact_bond_yield(T: float) -> float:    
    A = (1 - np.exp(-a * T)) / a
    B = (b - 0.5 * sigma**2 / a**2) * (A - T) - sigma**2 * A**2 / (4 * a)
    Z = np.exp(-A * r0 + B)
    y = - np.log(Z) / T
    
    return y
    

if __name__ == '__main__':
    TT = np.arange(1, 10, 1)
    
    tik = time.time()

    # Multi-core version    
    pool = Pool(6)
    yy = pool.map(bond_yield, TT)
    
    # Single-core version
    #yy = np.array(list(map(bond_yield, TT)))
    
    tok = time.time()
    
    print('Time = ', tok-tik)
    
    yy2 = list(map(exact_bond_yield, TT))

    plt.plot(TT, yy)
    plt.plot(TT, yy2, 'k:')
    plt.ylim(0, 0.04)
    plt.show()
