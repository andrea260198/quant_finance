#!/usr/bin/python3
#
# Title:
# Author:
# Python: 3.8

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
from option_pricing.options import EuropeanOption


if __name__ == '__main__':
    expiry: float = 1.0
    european_option = EuropeanOption(T=expiry, r=0.05, S_0=100.0, sigma=0.20, K=100.0)

    xx = np.arange(1000, 15_000, 1000)
    
    tik = time.time()
    
    # Multi-core
    pool = Pool(6)
    yy = np.array(list(pool.map(european_option.price_approx, xx)))
    
    # Mono-core
    #yy = np.array(list(map(price_approx, xx)))
    
    tok = time.time()
    
    print('Time = ', tok-tik, 's')
    
    dt = expiry/(xx-1)
    error = yy - european_option.price_exact()
    plt.loglog(dt, error, 'k')
    plt.ylabel('error [\$]')
    plt.xlabel('timestep [year]')
    plt.show()
