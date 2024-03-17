import numpy as np

from option_pricing.abstract_option import AbstractOption
from option_pricing.european_options import EuropeanPutOption


class EuropeanPutOptionBarrierIn(AbstractOption):
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
        super().__init__()
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
            self._expiry,
            self._r,
            self._S0,
            self._sigma,
            self._K
        )

        V = european_put_option.price_approx(N) - european_put_option_barrier_out.price_approx(N)

        self._price = V
        return self._price


class EuropeanPutOptionBarrierOut(AbstractOption):
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
        super().__init__()
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

        self._price = V[0]
        return self._price