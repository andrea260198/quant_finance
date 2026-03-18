#!/usr/bin/python3
#
# Title:
# Author:
# Python: 3.8

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
from core.option_pricing.european_options import EuropeanCallOption
import psutil


def main():
    expiry: float = 1.0  # [years]
    european_option = EuropeanCallOption(T=expiry, r=0.05, S_0=100.0, sigma=0.20, strike=100.0)

    xx = np.arange(1000, 15_000, 1000)  # np.array with number of discretizetion steps

    tik = time.time()

    # Multi-core
    n_cores = psutil.cpu_count(logical=False)
    pool = Pool(n_cores)
    yy = np.array(list(pool.map(european_option.price_approx, xx)))

    # Mono-core
    # yy = np.array(list(map(price_approx, xx)))

    tok = time.time()

    print('Time = ', tok - tik, 's')

    dt = expiry / (xx - 1)
    error = yy - european_option.price_exact()
    plt.loglog(dt, error, 'k')
    plt.ylabel('error [\$]')
    plt.xlabel('timestep [year]')
    plt.show()


if __name__ == '__main__':
    main()
