from typing import Literal

import numpy as np
import numpy.typing as npt
from statistics import NormalDist
from tqdm import tqdm
from abc import abstractmethod
from overrides import override
from core.option_pricing.abstract_option import AbstractOption, ExactTrait, ApproxTrait, MonteCarloTrait


class AbstractEuropeanVanillaOption(AbstractOption, ApproxTrait, MonteCarloTrait):
    T: float
    r: float
    S_0: float
    sigma: float

    @override
    def price_approx(self, N: int) -> float:
        """
        Approximate option price with a binomial tree.
        """
        T = self.T
        r = self.r
        S_0 = self.S_0
        sigma = self.sigma

        dt = T / (N-1)
        V = np.zeros((N, 1))
        S_T = np.zeros((N, 1))

        u = np.exp(sigma * np.sqrt(dt))
        v = 1 / u
        p = (np.exp(r * dt) - v) / (u - v)
        q = 1 - p

        S_T = np.array([[S_0 * u**(N-k-1) * v**k] for k in range(N)])

        V_new = self._calc_payoff(S_T)

        for j in tqdm(range(N-1, 0, -1)):
            V = V_new
            V_new[:j, 0] = np.exp(-r * dt) * (p * V[:j, 0] + q * V[1:j+1, 0])

        V_appr: float = V_new[0, 0]
        return V_appr

    def price_mc_approx(self, M: int, dt: float) -> float:
        """
        Approximate option price with Monte Carlo method.
        """

        T = self.T
        S_0 = self.S_0
        r = self.r
        sigma = self.sigma

        S = S_0 * np.ones((M, 1))

        for t in tqdm(np.arange(0, T, dt)):
            Z = np.random.normal(0, 1, (M, 1))
            dX = Z * np.sqrt(dt)
            dS = r * S * dt + sigma * S * dX
            S += dS

        payoff = self._calc_payoff(S)
        avg_payoff = np.mean(payoff)
        price: float = avg_payoff * np.exp(-r * T)
        return price

    @abstractmethod
    def _calc_payoff(self, S_T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass


class EuropeanCallOption(AbstractEuropeanVanillaOption, ExactTrait):
    type: Literal["EuropeanCallOption"]
    strike: float

    def price_exact(self) -> float:
        T = self.T
        r = self.r
        S_0 = self.S_0
        sigma = self.sigma
        K = self.strike

        d1 = ((r + 0.5 * sigma**2) * T - np.log(K / S_0)) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        def Phi(x: float) -> float:
            return NormalDist(0, 1).cdf(x)

        V_exact: float = S_0 * Phi(d1) - K * np.exp(-r * T) * Phi(d2)
        return V_exact

    @override
    def _calc_payoff(self, S_T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        K = self.strike
        return np.maximum(S_T - K, 0)  # Get element-wise max value


class EuropeanPutOption(AbstractEuropeanVanillaOption):
    type: str = "EuropeanPutOption"
    strike: float

    @override
    def _calc_payoff(self, S_T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        K = self.strike
        return np.maximum(K - S_T, 0)  # Get element-wise max value


class EuropeanDigitalCallOption(AbstractEuropeanVanillaOption):
    strike: float

    def price_exact(self) -> float:
        raise NotImplementedError

    @override
    def _calc_payoff(self, S_T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        K = self.strike
        mask = S_T - K > 0
        payoff = mask.astype(np.float64)
        return payoff
