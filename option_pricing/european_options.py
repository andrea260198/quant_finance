import numpy as np
import numpy.typing as npt
from statistics import NormalDist
from tqdm import tqdm
from abc import abstractmethod
from overrides import override
from option_pricing.abstract_option import AbstractOption


class AbstractEuropeanVanillaOption(AbstractOption):
    def __init__(
            self,
            T: float,
            r: float,
            S_0: float,
            sigma: float,
    ):
        super().__init__()
        self.T: float = T  # 1
        self.r: float = r  # 0.05
        self.S_0: float = S_0  # 100
        self.sigma: float = sigma  # 0.20

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

        for j in tqdm(range(N-1, 0, -1)):
            V = V_new
            V_new[:j, 0] = np.exp(-r * dt) * (p * V[:j, 0] + q * V[1:j+1, 0])

        V_appr: float = V_new[0, 0]
        self._price = V_appr
        return self._price

    @abstractmethod
    def _calc_payoff(self, S_T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass


class EuropeanCallOption(AbstractEuropeanVanillaOption):
    def __init__(
            self,
            T: float,
            r: float,
            S_0: float,
            sigma: float,
            K: float
    ):
        super().__init__(
            T,
            r,
            S_0,
            sigma
        )
        self._strike = K

    def price_exact(self) -> float:
        T = self.T
        r = self.r
        S_0 = self.S_0
        sigma = self.sigma
        K = self._strike

        d1 = ((r + 0.5 * sigma**2) * T - np.log(K / S_0)) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        def Phi(x: float) -> float:
            return NormalDist(0, 1).cdf(x)

        V_exact: float = S_0 * Phi(d1) - K * np.exp(-r * T) * Phi(d2)
        self._price = V_exact
        return self._price

    @override
    def _calc_payoff(self, S_T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        K = self._strike
        return np.maximum(S_T - K, 0)  # Get element-wise max value


class EuropeanPutOption(AbstractEuropeanVanillaOption):
    def __init__(
            self,
            T: float,
            r: float,
            S_0: float,
            sigma: float,
            K: float
    ):
        super().__init__(
            T,
            r,
            S_0,
            sigma
        )
        self._strike = K

    @override
    def _calc_payoff(self, S_T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        K = self._strike
        return np.maximum(K - S_T, 0)  # Get element-wise max value


class EuropeanDigitalCallOption(AbstractEuropeanVanillaOption):
    def __init__(
            self,
            T: float,
            r: float,
            S_0: float,
            sigma: float,
            K: float
    ):
        super().__init__(
            T,
            r,
            S_0,
            sigma
        )
        self._strike = K

    def price_exact(self) -> float:
        raise NotImplementedError

    @override
    def _calc_payoff(self, S_T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        K = self._strike
        mask = S_T - K > 0
        payoff = mask.astype(np.float64)
        return payoff
