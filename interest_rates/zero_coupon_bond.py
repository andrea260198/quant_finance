import numpy as np
from interest_rates.short_rate_models import ShortRateModel
from dataclasses import dataclass
from scipy.stats import norm, qmc
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ZeroCouponBond:
    T: float
    short_rate_model: ShortRateModel
    M: int = 10_000  # Monte Carlo simulation sample size

    def calc_approx_yield(self) -> float:
        """
        Calculate yield using an approximation of bond price using Monte Carlo simulation
        :param T:
        :return:
        """
        model: ShortRateModel = self.short_rate_model
        # Calculate mean as approx of zero-coupon bond price
        Z = np.mean([np.exp(-model.integrate_r_dt(self.T)) for k in range(self.M)])
        # Calculate yield
        y: float = - np.log(Z) / self.T
        print("Zero-coupon bond of maturity T = {} has value Z = {:.3f}".format(self.T, Z))
        return y

    def calc_qmc_approx_yield(self) -> float:
        T = self.T
        dt = self.short_rate_model._dt
        a = self.short_rate_model._a
        b = self.short_rate_model._b
        r0 = self.short_rate_model._r0
        sigma = self.short_rate_model._sigma

        def calculate_random_walks(M: int) -> npt.NDArray[np.float64]:
            N = int(T // dt)
            U = get_low_discr_sample(N, M)

            def integrate(U: npt.NDArray[np.float64], dt: float):
                # We want to integrate the Vasicek SDE:
                # dr = a * (b - r) * dt + sigma * r * dW
                dW = np.sqrt(12) * (U - 0.5) * np.sqrt(dt)

                r_t = r0 * np.ones((1, M))
                for k, t in enumerate(np.arange(0, T, dt)):
                    r_t += a * (b - r_t) * dt + sigma * r_t * dW[k, :]

                r_T = r_t
                return r_T

            S_T = integrate(U, dt)
            return S_T

        Z = np.mean(calculate_random_walks(self.M))
        # Calculate yield
        y: float = - np.log(Z) / self.T
        print("Zero-coupon bond of maturity T = {} has value Z = {:.3f}".format(self.T, Z))
        return y

    def calc_exact_yield(self) -> float:
        return self.short_rate_model.calc_exact_yield(self.T)


def get_low_discr_sample(N: int, M: int) -> npt.NDArray[np.float64]:
    sampler = qmc.Sobol(d=N, scramble=False)
    sobol_sequences = sampler.random(M)
    return sobol_sequences.transpose()


