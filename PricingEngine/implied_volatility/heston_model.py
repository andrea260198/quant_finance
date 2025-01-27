import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from interest_rates.short_rate_models import get_low_discr_sample
from option_pricing.abstract_option import AbstractOption
from overrides import override
from scipy.special import ndtri


class HestonEuropeanCallOption(AbstractOption):
    type: str = "HestonEuropeanCallOption"
    T: float
    r: float
    S_0: float
    sigma: float
    strike: float

    @override
    def price_approx(self, N: int) -> float:
        return self.price_mc_approx(N, 0.01)

    def price_mc_approx(self, M: int, dt: float) -> float:
        """
        Approximate option price with Monte Carlo method.
        """

        T = self.T
        S_0 = self.S_0
        r = self.r
        var_0 = self.sigma**2

        #  Variance CIR process parameters
        # NOTE: The following condition must be satisfied:
        #       2 * k * theta >= s**2
        k = 0.20
        theta = 0.20
        s = 0.10

        S = S_0 * np.ones((M, 1))
        var = var_0 * np.ones((M, 1))

        rho = 0.5  # TODO: set correct value

        TT = np.arange(0, T, dt)

        U = get_low_discr_sample(len(TT)*2, M)
        U1, U2 = U[0:len(TT), :], U[len(TT):, :]

        for i, t in enumerate(TT):
            cov = np.array([[1, rho], [rho, 1]])

            W1, W2 = U1[[i], :], U2[[i], :]

            Z1, Z2 = np.dot(cov, ndtri(np.concatenate([W1, W2], axis=0)))

            Z1 = np.expand_dims(Z1, axis=1)
            Z2 = np.expand_dims(Z2, axis=1)
            dX = Z1 * np.sqrt(dt)
            dY = Z2 * np.sqrt(dt)
            dS = r * S * dt + np.sqrt(var) * S * dX
            S += dS
            d_var = k * (theta - var) * dt + s * np.sqrt(var) * dY
            var += d_var

        payoff = self._calc_payoff(S)

        avg_payoff = np.mean(payoff)

        price = avg_payoff  * np.exp(-r * T)

        return float(price)

    def _calc_payoff(self, S_T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        K = self.strike
        return np.maximum(S_T - K, 0)  # Get element-wise max value











