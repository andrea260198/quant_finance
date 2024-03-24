import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from option_pricing.american_options import AmericanCallOption
from option_pricing.european_options import EuropeanCallOption

if __name__ == '__main__':
    K_vec = np.arange(50, 150, 1)
    T_vec = np.arange(0.1, 1.1, 0.01)
    S0 = 100
    r = 0.05
    sigma = 0.20

    K_mg, T_mg = np.meshgrid(K_vec, T_vec)  # type: ignore

    V_mg = np.array([[AmericanCallOption(T, r, S0, sigma, K).price_approx(500) for K in K_vec] for T in T_vec])

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


    surf = ax.plot_surface(K_mg, T_mg, V_mg)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Expiry")
    ax.set_zlabel("Price")

    plt.show()