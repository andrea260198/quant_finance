import numpy as np
from abc import ABC, abstractmethod
from overrides import override
from dataclasses import dataclass
import numpy.typing as npt
from scipy.stats import qmc


@dataclass(frozen=True)
class ModelParameters:
    dt: float = 0.01
    a: float = 0.1
    b: float = 0.07
    r0: float = 0.02
    sigma: float = 0.02


class ShortRateModel(ABC):
    """
    This abstract class contains all the common methods between short interest rate models.
    """
    def __init__(
            self,
            model_parameters: ModelParameters
    ):
        self._dt: float = model_parameters.dt
        self._a: float = model_parameters.a
        self._b: float = model_parameters.b
        self._r0: float = model_parameters.r0
        self._sigma: float = model_parameters.sigma
        seed = 1
        self._rng = np.random.default_rng(seed)

    def _dX(self) -> float:
        Z: float = np.random.normal()  # Standard normal r.v.
        dX: float = Z * np.sqrt(self._dt)  # X(t) is a Brownian motion
        # bernoulli_rv = self._rng.integers(0, 2)  # Slower
        # dX = (2 * bernoulli_rv - 1) * np.sqrt(self._dt)
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

    @abstractmethod
    def approximate_integral_with_qmc(self, T: float, M: int) -> npt.NDArray[np.float64]:
        ...

    @abstractmethod
    def calc_exact_yield(self, T) -> float:
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

    @override
    def approximate_integral_with_qmc(self, T: float, M: int) -> npt.NDArray[np.float64]:
        dt = self._dt
        a = self._a
        b = self._b
        r0 = self._r0
        sigma = self._sigma

        N = int(T // dt)
        #U = get_low_discr_sample(N, M)
        U = np.random.uniform(size=(N, M))

        def integrate(U: npt.NDArray[np.float64], dt: float):
            # We want to integrate the Vasicek SDE:
            # dr = a * (b - r) * dt + sigma * r * dW
            dW = np.sqrt(12) * (U - 0.5) * np.sqrt(dt)

            r_t = r0 * np.ones((1, M))
            integral_t = 0
            for k in range(N):
                integral_t += r_t * dt
                r_t += a * (b - r_t) * dt + sigma * dW[k, :]

            integral_T = integral_t
            return integral_T

        integral_T = integrate(U, dt)
        return integral_T

    def calc_exact_yield(self, T: int) -> float:
        """
        # Calculate yield using exact bond pricing formula for Vasicek model
        :param T:
        :return:
        """
        A = (1 - np.exp(-self._a * T)) / self._a
        B = (self._b - 0.5 * self._sigma**2 / self._a**2) * (A - T) - self._sigma**2 * A**2 / (4 * self._a)
        Z = np.exp(-A * self._r0 + B)
        y: float = - np.log(Z) / T
        return y


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


def get_low_discr_sample(N: int, M: int) -> npt.NDArray[np.float64]:
    sampler = qmc.Sobol(d=N, scramble=False)
    sobol_sequences = sampler.random(M)
    return sobol_sequences.transpose()