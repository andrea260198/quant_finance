import numpy as np
from statistics import NormalDist
from tqdm import tqdm
from abc import ABC, abstractmethod
from overrides import override


class IOption(ABC):
    @abstractmethod
    def price_approx(self, N: int) -> float:
        ...


class EuropeanOption(IOption):
    def __init__(
            self,
            T: float,
            r: float,
            S_0: float,
            sigma: float,
            K: float
    ):
        self.T: float = T  # 1
        self.r: float = r  # 0.05
        self.S_0: float = S_0  # 100
        self.sigma: float = sigma  # 0.20
        self.K: float = K  # 100

    @override
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

        V_appr: float = V_new[0, 0]
        return V_appr


    def price_exact(self) -> float:
        T = self.T
        r = self.r
        S_0 = self.S_0
        sigma = self.sigma
        K = self.K

        d1 = ((r + 0.5 * sigma**2) * T - np.log(K / S_0)) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        def Phi(x: float) -> float:
            return NormalDist(0, 1).cdf(x)

        V_exact: float = S_0 * Phi(d1) - K * np.exp(-r * T) * Phi(d2)
        return V_exact