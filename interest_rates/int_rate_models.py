import numpy as np
from abc import ABC, abstractmethod
from overrides import override


class InterestRateModel(ABC):
    @abstractmethod
    def integrate_r_dt(self, T: float) -> float:
        ...


class VasicekModel(InterestRateModel):
    def __init__(
            self,
            dt: float,
            a: float,
            b: float,
            r0: float,
            sigma: float
    ):
        self.dt: float = dt
        self.a: float = a
        self.b: float = b
        self.r0: float = r0
        self.sigma: float = sigma
        Z = lambda: np.random.normal()  # Standard normal r.v.
        self.dX = lambda: Z() * np.sqrt(dt)  # X(t) is a Brownian motion

    @override
    def integrate_r_dt(self, T: float) -> float:
        a = self.a
        b = self.b
        dt = self.dt
        sigma = self.sigma
        dX = self.dX

        N = int(T / self.dt)

        r = np.zeros(N)

        r[0] = self.r0

        for k in range(N - 1):
            # Vasicek model
            dr = a * (b - r[k]) * dt + sigma * dX()
            r[k + 1] = r[k] + dr

        integral = sum(r * dt)
        return integral


class CoxIngersolRossModel(InterestRateModel):
    def __init__(
            self,
            dt: float,
            a: float,
            b: float,
            r0: float,
            sigma: float
    ):
        self.dt: float = dt
        self.a: float = a
        self.b: float = b
        self.r0: float = r0
        self.sigma: float = sigma
        Z = lambda: np.random.normal()  # Standard normal r.v.
        self.dX = lambda: Z() * np.sqrt(dt)  # X(t) is a Brownian motion

    @override
    def integrate_r_dt(self, T: float) -> float:
        a = self.a
        b = self.b
        dt = self.dt
        sigma = self.sigma
        dX = self.dX

        N = int(T / self.dt)

        r = np.zeros(N)

        r[0] = self.r0

        for k in range(N - 1):
            # CIR model
            dr = a * (b - r[k]) * dt + self.sigma * np.sqrt(r[k]) * dX()
            r[k + 1] = r[k] + dr

        integral = sum(r * dt)
        return integral