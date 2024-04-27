import itertools

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from option_pricing.abstract_option import AbstractOption
import matplotlib.pyplot as plt

from scipy.optimize import root

from option_pricing.european_options import EuropeanCallOption


class HestonEuropeanCallOption(AbstractOption):
    def __init__(
            self,
            T: float,
            r: float,
            S_0: float,
            sigma: float,
            K: float
    ):
        super().__init__()
        self.T: float = T
        self.r: float = r
        self.S_0: float = S_0
        self.sigma: float = sigma
        self._strike = K

    def price_approx(self, N: int) -> float:
        self.price_mc_approx(N, 0.01)

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

        for t in tqdm(np.arange(0, T, dt)):
            mean = [0, 0]
            cov = [[1, rho], [rho, 1]]
            Z1, Z2 = np.random.multivariate_normal(mean, cov, M).T
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

        self._price = avg_payoff  * np.exp(-r * T)

        return self._price

    def _calc_payoff(self, S_T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        K = self._strike
        return np.maximum(S_T - K, 0)  # Get element-wise max value


def calc_cartesian_product(x, y):
    I = len(x)
    J = len(y)
    X = [[(x[i], y[j]) for j in range(J)] for i in range(I)]
    return X


def heston_price_surface():
    KK = np.arange(50, 150, 1)
    TT = np.arange(0.1, 3, 0.1)
    K_mg, T_mg = np.meshgrid(KK, TT, indexing="ij")

    sigma = 0.10
    S_0 = 100
    r = 0.05

    V_mg = np.zeros((len(KK), len(TT)))

    I = len(KK)
    J = len(TT)

    M = 10_000
    dt = 0.01

    for i in range(I):
        for j in range(J):
            V_mg[i, j] = HestonEuropeanCallOption(TT[j], r, S_0, sigma, KK[i]).price_mc_approx(M, dt)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


    surf = ax.plot_surface(K_mg, T_mg, V_mg)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Expiry")
    ax.set_zlabel("Price")

    plt.show()

def volatility_surface():
    KK = np.arange(50, 150, 1)
    TT = np.arange(0.1, 2, 0.1)
    K_mg, T_mg = np.meshgrid(KK, TT, indexing="ij")

    sigma_0 = 0.10
    S_0 = 100
    r = 0.05

    sigma_mg = np.zeros((len(KK), len(TT)))

    I = len(KK)
    J = len(TT)

    M = 10_000
    dt = 0.01

    #for (i, j) in itertools.product(range(I), range(J)):
    #   pass

    for i in range(I):
        for j in range(J):
            V = HestonEuropeanCallOption(TT[j], r, S_0, sigma_0, KK[i]).price_mc_approx(M, dt)

            def fun(sigma):
                return EuropeanCallOption(TT[j], r, S_0, sigma, KK[i]).price_exact() - V

            sol = root(fun, sigma_0)
            # Negative volatility results are not accepted
            sigma_mg[i, j] = sol.x[0] if sol.x[0] > 0 else np.nan


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


    surf = ax.plot_surface(K_mg, T_mg, sigma_mg)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Expiry")
    ax.set_zlabel("Implied volatility")

    plt.show()








