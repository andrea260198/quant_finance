#!/usr/bin/python3
#
# Title:
# Author:
# Python: 3.8

import numpy as np
from option_pricing.options import EuropeanOption


def test_european_option() -> None:
    expiry: float = 1.0  # [years]
    european_option = EuropeanOption(T=expiry, r=0.05, S_0=100.0, sigma=0.20, K=100.0)
    timesteps = 1000
    assert abs(european_option.price_approx(timesteps) - european_option.price_exact()) < 0.01
