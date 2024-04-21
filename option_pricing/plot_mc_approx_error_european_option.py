from option_pricing.european_options import EuropeanCallOption
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    option = EuropeanCallOption(
        T=10,
        r=0.20,
        S_0=100,
        sigma=0.20,
        K=100
    )

    MM = [10, 100, 1000]#, 10_000, 100_000]
    std_err_list = []



    for M in MM:
        errors = np.array([option.price_mc_approx(M, 0.01) for k in range(100)]) - option.price_exact()
        std_err = np.std(errors)
        print(std_err)
        std_err_list += [std_err]

    plt.loglog(MM, std_err_list, 'k')
    plt.ylabel('std err [$]')
    plt.xlabel('M')
    plt.show()

    """
    Conclusions:
    std_err = O(1 / sqrt(M))
    """