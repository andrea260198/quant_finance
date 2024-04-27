from implied_volatility.heston_model import HestonEuropeanCallOption
import matplotlib.pyplot as plt
from scipy.optimize import root
from option_pricing.european_options import EuropeanCallOption
import numpy as np


def plot_heston_price_surface():
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

def plot_volatility_surface():
    KK = np.arange(80, 120, 1)
    TT = np.arange(0.1, 1, 0.1)
    K_mg, T_mg = np.meshgrid(KK, TT, indexing="ij")

    sigma_0 = 0.10
    S_0 = 100
    r = 0.05

    sigma_mg = np.zeros((len(KK), len(TT)))

    I = len(KK)
    J = len(TT)

    M = 100_000
    dt = 0.001

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


if __name__ == '__main__':
    plot_volatility_surface()