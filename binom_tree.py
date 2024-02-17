#!/usr/bin/python3
#
# Title:
# Author:
# Python: 3.8

import numpy as np
from statistics import NormalDist
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
from abc import ABC, abstractmethod


class Option(ABC):
    @abstractmethod
    def price_approx(self):
        ...


class EuropeanOption(Option):
    def __init__(
            self,
            T: float,
            r: float,
            S_0: float,
            sigma: float,
            K: float):
        self.T: float = T  # 1
        self.r: float = r  # 0.05
        self.S_0: float = S_0  # 100
        self.sigma: float = sigma  # 0.20
        self.K: float = K  # 100


    def price_approx(self, N: int) -> float:
        T = self.T
        r = self.r
        S_0 = self.S_0
        sigma = self.sigma
        K = self.K

        dt = T / (N-1)
        V = np.zeros((N, 1))
        S_T = np.zeros((N, 1))

        u = np.exp(sigma * np.sqrt(dt))
        v = 1 / u
        p = (np.exp(r * dt) - v) / (u - v)
        q = 1 - p

        S_T = np.array([[S_0 * u**(N-k-1) * v**k] for k in range(N)])

        V_new = np.maximum(S_T - K, 0)  # Get element-wise max value

        for j in tqdm(range(N-1, 0, -1)):
            V = V_new
            V_new[:j, 0] = np.exp(-r * dt) * (p * V[:j, 0] + q * V[1:j+1, 0])

        V_appr = V_new[0, 0]
        return V_appr


    def price_exact(self) -> float:
        T = self.T
        r = self.r
        S_0 = self.S_0
        sigma = self.sigma
        K = self.K

        d1 = ((r + 0.5 * sigma**2) * T - np.log(K / S_0)) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        Phi = lambda x: NormalDist(0, 1).cdf(x)
        V_exact = S_0 * Phi(d1) - K * np.exp(-r * T) * Phi(d2)
        return V_exact


if __name__ == '__main__':
    expiry: float = 1.0
    european_option = EuropeanOption(T=expiry, r=0.05, S_0=100.0, sigma=0.20, K=100.0)

    xx = np.arange(1000, 15_000, 1000)
    
    tik = time.time()
    
    # Multi-core
    pool = Pool(6)
    yy = np.array(list(pool.map(european_option.price_approx, xx)))
    
    # Mono-core
    #yy = np.array(list(map(price_approx, xx)))
    
    tok = time.time()
    
    print('Time = ', tok-tik, 's')
    
    dt = expiry/(xx-1)
    error = yy - european_option.price_exact()
    plt.loglog(dt, error, 'k')
    plt.ylabel('error [\$]')
    plt.xlabel('timestep [year]')
    plt.show()
