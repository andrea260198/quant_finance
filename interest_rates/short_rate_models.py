import numpy as np
from abc import ABC, abstractmethod
from overrides import override


class ShortRateModel(ABC):
    """
    This abstract class contains all the common methods between short interest rate models.
    """
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

    def _dX(self) -> float:
        Z: float = np.random.normal()  # Standard normal r.v.
        dX: float = Z * np.sqrt(self._dt)  # X(t) is a Brownian motion
        return dX

    def integrate_r_dt(self, T: float) -> float:
        dt = self._dt
        N = int(T / dt)
        r = np.zeros(N)
        r[0] = self._r0
        for k in range(N - 1):
            dr = self.calc_dr(r[k])
            r[k + 1] = r[k] + dr

        integral: float = sum(r * dt)
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
        """
        a = self._a
        b = self._b
        dt = self._dt
        sigma = self._sigma
        dX = self._dX

        dr: float = a * (b - r) * dt + sigma * np.sqrt(r) * dX()
        return dr
