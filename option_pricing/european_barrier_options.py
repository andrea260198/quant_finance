import numpy as np

from option_pricing.abstract_option import AbstractOption
from option_pricing.european_options import EuropeanPutOption


class EuropeanPutOptionBarrierIn(AbstractOption):
    r: float
    sigma: float
    S0: float
    div: float
    expiry: float
    strike: float
    beta: float

    def price_approx(self, N: int) -> float:
        european_put_option_barrier_out = EuropeanPutOptionBarrierOut(
            r=self.r,
            sigma=self.sigma,
            S0=self.S0,
            div=self.div,
            expiry=self.expiry,
            strike=self.strike,
            beta=self.beta
        )

        european_put_option = EuropeanPutOption(
            expiry=self.expiry,
            r=self.r,
            S0=self.S0,
            sigma=self.sigma,
            strike=self.strike
        )

        V = european_put_option.price_approx(N) - european_put_option_barrier_out.price_approx(N)
        return V


class EuropeanPutOptionBarrierOut(AbstractOption):
    r: float
    sigma: float
    S0: float
    div: float
    expiry: float
    strike: float
    beta: float

    def price_approx(self, N: int) -> float:
        sigma = self.sigma
        S0 = self.S0
        beta = self.beta
        div = self.div
        r = self.r
        dt = self.expiry / N
        K = self.strike

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