from option_pricing.european_options import EuropeanCallOption
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool


def calc_std_error(M: int) -> float:
    option = EuropeanCallOption(
        T=10,
        r=0.20,
        S_0=100,
        sigma=0.20,
        strike=100
    )

    errors = np.array([option.price_mc_approx(M, 0.01) for k in range(100)]) - option.price_exact()
    std_err = np.std(errors)
    print(std_err)
    return float(std_err)


if __name__ == '__main__':
    MM = [10, 100, 1000, 10_000, 100_000]

    pool = Pool(6)

    std_err_list = list(pool.map(calc_std_error, MM))

    plt.loglog(MM, std_err_list, 'k')
    plt.ylabel('std err [$]')
    plt.xlabel('M')
    plt.show()

    """
    Conclusions:
    std_err = O(1 / sqrt(M))
    """