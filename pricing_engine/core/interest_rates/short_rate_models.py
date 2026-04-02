from typing import Literal

import numpy as np
from abc import ABC, abstractmethod
from overrides import override
import numpy.typing as npt
from scipy.stats import qmc

from pricing_engine.support.quant_dataclass import ImmutableDataclass


class ShortRateModel(ImmutableDataclass, ABC):
    dt: float
    a: float
    b: float
    r0: float
    sigma: float
    #seed = 1
    #self._rng = np.random.default_rng(seed)

    def _dX(self) -> float:
        Z: float = np.random.normal()  # Standard normal r.v.
        dX: float = Z * np.sqrt(self.dt)  # X(t) is a Brownian motion
        # bernoulli_rv = self._rng.integers(0, 2)  # Slower
        # dX = (2 * bernoulli_rv - 1) * np.sqrt(self._dt)
        return dX

    def integrate_r_dt(self, T: float) -> float:
        dt = self.dt
        N = int(T / dt)
        r = np.zeros(N)
        r[0] = self.r0
        for k in range(N - 1):
            dr = self.calc_dr(r[k])
            r[k + 1] = r[k] + dr

        integral: float = sum(r * dt)
        return integral

    @abstractmethod
    def calc_dr(self, r: float) -> float:
        ...

    @abstractmethod
    def approximate_integral_with_qmc(self, T: float, M: int) -> npt.NDArray[np.float64]:
        ...

    @abstractmethod
    def calc_exact_yield(self, T: float) -> float:
        ...

class VasicekModel(ShortRateModel):
    type: Literal["VasicekModel"] = "VasicekModel"

    @override
    def calc_dr(self, r: float) -> float:
        """
        Return differential dr of Vasicek model SDE
        dr = a * (b - r[k]) * dt + sigma * dX()
        """
        a = self.a
        b = self.b
        dt = self.dt
        sigma = self.sigma
        dX = self._dX

        dr = a * (b - r) * dt + sigma * dX()
        return dr

    @override
    def approximate_integral_with_qmc(self, T: float, M: int) -> npt.NDArray[np.float64]:
        dt = self.dt
        a = self.a
        b = self.b
        r0 = self.r0
        sigma = self.sigma

        N = int(T // dt)
        U = get_low_discr_sample(N, M)
        #U = np.random.uniform(size=(N, M))

        def integrate(U: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
            # We want to integrate the Vasicek SDE:
            # dr = a * (b - r) * dt + sigma * r * dW
            dW = np.sqrt(12) * (U - 0.5) * np.sqrt(dt)

            r_t = r0 * np.ones((1, M))
            integral_t = np.zeros((1, M))
            for k in range(N):
                integral_t += r_t * dt
                r_t += a * (b - r_t) * dt + sigma * dW[k, :]

            integral_T = integral_t
            return integral_T

        integral_T = integrate(U, dt)
        return integral_T

    def calc_exact_yield(self, T: float) -> float:
        """
        # Calculate yield using exact bond pricing formula for Vasicek model
        :param T:
        :return:
        """
        A = (1 - np.exp(-self.a * T)) / self.a
        B = (self.b - 0.5 * self.sigma ** 2 / self.a ** 2) * (A - T) - self.sigma ** 2 * A ** 2 / (4 * self.a)
        Z = np.exp(-A * self.r0 + B)
        y: float = - np.log(Z) / T
        return y


class CoxIngersolRossModel(ShortRateModel):
    type: Literal["CoxIngersolRossModel"] = "CoxIngersolRossModel"

    @override
    def calc_dr(self, r: float) -> float:
        """
        Return differential dr of Cox-Ingersoll-Ross model SDE
        dr = a * (b - r[k]) * dt + sigma * dX()
        """
        a = self.a
        b = self.b
        dt = self.dt
        sigma = self.sigma
        dX = self._dX

        dr: float = a * (b - r) * dt + sigma * np.sqrt(r) * dX()
        return dr

    def approximate_integral_with_qmc(self, T: float, M: int) -> npt.NDArray[np.float64]:
        raise NotImplementedError()  # TODO: Implement

    def calc_exact_yield(self, T: float) -> float:
        raise NotImplementedError()  # TODO: Implement


def get_low_discr_sample(N: int, M: int) -> npt.NDArray[np.float64]:
    sampler = qmc.Sobol(d=N, scramble=True)
    sobol_sequences = np.array(sampler.random(M))
    return sobol_sequences.transpose()