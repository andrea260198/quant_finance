from abc import abstractmethod

import numpy as np
import numpy.typing as npt
from overrides import override
from tqdm import tqdm

from core.option_pricing.abstract_option import AbstractOption, ApproxTrait


class AbstractAmericanVanillaOption(AbstractOption, ApproxTrait):
    T: float
    r: float
    S_0: float
    sigma: float

    @override
    def price_approx(self, N: int) -> float:
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
        S_new = S_T / u

        for j in tqdm(range(N-1, 0, -1)):
            V = V_new
            S = S_new

            V_new[:j, 0] = np.maximum(
                np.exp(-r * dt) * (p * V[:j, 0] + q * V[1:j+1, 0]),
                self._calc_payoff(S[:j, 0])
            )
            S_new[:j, 0] = S[:j, 0] / u

        V_appr: float = V_new[0, 0]
        return V_appr

    @abstractmethod
    def _calc_payoff(self, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass


class AmericanCallOption(AbstractAmericanVanillaOption):
    type: str = "AmericanCallOption"
    strike: float

    @override
    def _calc_payoff(self, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        K = self.strike
        return np.maximum(S - K, 0)  # Get element-wise max value


class AmericanPutOption(AbstractAmericanVanillaOption):
    type: str = "AmericanPutOption"
    strike: float

    @override
    def _calc_payoff(self, S: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        K = self.strike
        return np.maximum(K - S, 0)  # Get element-wise max value