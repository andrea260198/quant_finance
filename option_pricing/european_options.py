import numpy as np
import numpy.typing as npt
from statistics import NormalDist
from tqdm import tqdm
from abc import ABC, abstractmethod
from overrides import override


class IOption(ABC):
    @abstractmethod
    def price_approx(self, N: int) -> float:
        pass


class AbstractEuropeanOption(IOption):
    def __init__(
            self,
            T: float,
            r: float,
            S_0: float,
            sigma: float,
    ):
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
        return V_appr

    @abstractmethod
    def _calc_payoff(self, S_T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass


class EuropeanCallOption(AbstractEuropeanOption):
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
        return V_exact

    @override
    def _calc_payoff(self, S_T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        K = self._strike
        return np.maximum(S_T - K, 0)  # Get element-wise max value


class EuropeanPutOption(AbstractEuropeanOption):
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


class EuropeanDigitalCallOption(AbstractEuropeanOption):
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


class EuropeanPutOptionBarrierIn(IOption):
    def __init__(
            self,
            r: float,
            sigma: float,
            S0: float,
            div: float,
            expiry: float,
            K: float,
            beta: float
    ):
        self._r = r
        self._sigma = sigma
        self._S0 = S0
        self._div = div
        self._expiry = expiry
        self._K = K
        self._beta = beta

    def price_approx(self, N: int) -> float:
        european_put_option_barrier_out = EuropeanPutOptionBarrierOut(
            self._r,
            self._sigma,
            self._S0,
            self._div,
            self._expiry,
            self._K,
            self._beta
        )

        european_put_option = EuropeanPutOption(
            self._T,
            self._r,
            self._S_0,
            self._sigma,
            self._K
        )

        V = european_put_option.price_approx(N) - european_put_option_barrier_out.price_approx(N)

        return V


class EuropeanPutOptionBarrierOut(IOption):
    def __init__(
            self,
            r: float,
            sigma: float,
            S0: float,
            div: float,
            expiry: float,
            K: float,
            beta: float
    ):
        self._r = r
        self._sigma = sigma
        self._S0 = S0
        self._div = div
        self._expiry = expiry
        self._K = K
        self._beta = beta

    def price_approx(self, N: int) -> float:
        sigma = self._sigma
        S0 = self._S0
        beta = self._beta
        div = self._div
        r = self._r
        dt = self._expiry / N
        K = self._K

        u = np.exp(sigma * np.sqrt(dt))
        v = 1 / u
        p = (np.exp((r - div) * dt) - v) / (u - v)
        q = 1 - p

        S = [S0 * (u ** (N - k)) * (v ** (k)) for k in range(N + 1)]

        V: list[float] = []

        for ST in S:
            if ST > beta * S0:
                V = V + [max(K - ST, 0)]
            else:
                V = V + [0]
        while len(V) > 1:
            V_up = V[0:-1]
            V_down = V[1:]
            V = [(p * V_up[k] + q * V_down[k]) / (1 + r * dt) for k in range(len(V_up))]
            S = [S0 * (u ** (N - k)) * (v ** (k)) for k in range(len(V_up))]
            for ST in S:
                k = 0
                if ST <= beta * S0:
                    V[k] = 0
                k += 1

        return V[0]
