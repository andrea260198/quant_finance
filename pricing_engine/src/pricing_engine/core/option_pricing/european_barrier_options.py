from typing import Literal

import numpy as np

from core.contract import PricingMethod
from core.option_pricing.abstract_option import AbstractOption, ApproxTrait
from core.option_pricing.european_options import EuropeanPutOption


class EuropeanPutOptionBarrierIn(AbstractOption, ApproxTrait):
    r: float
    sigma: float
    S_0: float
    div: float
    T: float
    strike: float
    beta: float
    type: Literal["EuropeanPutOptionBarrierIn"] = "EuropeanPutOptionBarrierIn"

    def price_approx(self, N: int) -> float:
        european_put_option_barrier_out = EuropeanPutOptionBarrierOut(
            r=self.r,
            sigma=self.sigma,
            S_0=self.S_0,
            div=self.div,
            T=self.T,
            strike=self.strike,
            beta=self.beta,
        )

        european_put_option = EuropeanPutOption(
            T=self.T,
            r=self.r,
            S_0=self.S_0,
            sigma=self.sigma,
            strike=self.strike,
        )

        V = european_put_option.price_approx(N) - european_put_option_barrier_out.price_approx(N)
        return V


class EuropeanPutOptionBarrierOut(AbstractOption, ApproxTrait):
    r: float
    sigma: float
    S_0: float
    div: float
    T: float
    strike: float
    beta: float
    pricing_method: PricingMethod = PricingMethod.BINOMIAL_TREE
    type: Literal["EuropeanPutOptionBarrierOut"] = "EuropeanPutOptionBarrierOut"

    def price_approx(self, N: int) -> float:
        sigma = self.sigma
        S0 = self.S_0
        beta = self.beta
        div = self.div
        r = self.r
        dt = self.T / N
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
