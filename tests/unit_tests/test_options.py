import numpy as np
from option_pricing.options import EuropeanCallOption


def test_european_option() -> None:
    expiry: float = 1.0  # [years]
    european_call_option = EuropeanCallOption(T=expiry, r=0.05, S_0=100.0, sigma=0.20, K=100.0)
    timesteps = 1000
    assert abs(european_call_option.price_approx(timesteps) - european_call_option.price_exact()) < 0.01


def test_put_call_parity() -> None:
    expiry: float = 1.0  # [years]
    european_call_option = EuropeanCallOption(T=expiry, r=0.05, S_0=100.0, sigma=0.20, K=100.0)
    european_put_option = EuropeanCallOption(T=expiry, r=0.05, S_0=100.0, sigma=0.20, K=100.0)
    timesteps = 1000
    # Verify put-call parity
    assert abs(european_call_option.price_approx(timesteps) - european_put_option.price_approx(timesteps) - 100 + 100 * np.exp(-0.05 * expiry)) < 0.01
