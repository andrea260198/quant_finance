import numpy as np
from abc import ABC, abstractmethod
from overrides import override


class ShortRateModel(ABC):
    def __init__(
            self,
            dt: float,
            a: float,
            b: float,
            r0: float,
            sigma: float
    ):
        self._dt: float = dt
        self._a: float = a
        self._b: float = b
        self._r0: float = r0
        self._sigma: float = sigma
        Z = lambda: np.random.normal()  # Standard normal r.v.
        self._dX = lambda: Z() * np.sqrt(dt)  # X(t) is a Brownian motion

    def integrate_r_dt(self, T: float) -> float:
        dt = self._dt

        N = int(T / dt)

        r = np.zeros(N)

        r[0] = self._r0

        for k in range(N - 1):
            dr = self.calc_dr(r[k])
            r[k + 1] = r[k] + dr

        integral = sum(r * dt)
        return integral

    @abstractmethod
    def calc_dr(self, r: float) -> float:
        ...


class VasicekModel(ShortRateModel):
    @override
    def calc_dr(self, r: float) -> float:
        """
        Return differential dr of Vasicek model SDE
        dr = a * (b - r[k]) * dt + sigma * dX()
        :param r:
        :return:
        """
        a = self._a
        b = self._b
        dt = self._dt
        sigma = self._sigma
        dX = self._dX

        dr = a * (b - r) * dt + sigma * dX()
        return dr


class CoxIngersolRossModel(ShortRateModel):
    @override
    def calc_dr(self, r: float) -> float:
        """
        Return differential dr of Cox-Ingersoll-Ross model SDE
        dr = a * (b - r[k]) * dt + sigma * dX()
        :param r:
        :return:
        """
        a = self._a
        b = self._b
        dt = self._dt
        sigma = self._sigma
        dX = self._dX

        dr = a * (b - r) * dt + sigma * np.sqrt(r) * dX()
        return dr
