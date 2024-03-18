import numpy as np

from option_pricing.european_barrier_options import EuropeanPutOptionBarrierOut
from option_pricing.european_options import EuropeanCallOption, EuropeanPutOption


def test_european_option() -> None:
    expiry: float = 1.0  # [years]
    european_call_option = EuropeanCallOption(T=expiry, r=0.05, S_0=100.0, sigma=0.20, K=100.0)
    timesteps = 1000
    assert abs(european_call_option.price_approx(timesteps) - european_call_option.price_exact()) < 0.01


def test_put_call_parity() -> None:
    expiry: float = 1.0  # [years]
    european_call_option = EuropeanCallOption(T=expiry, r=0.05, S_0=100.0, sigma=0.20, K=100.0)
    european_put_option = EuropeanPutOption(T=expiry, r=0.05, S_0=100.0, sigma=0.20, K=100.0)
    timesteps = 1000

    european_call_option.price_approx(timesteps)
    european_put_option.price_approx(timesteps)

    call_price = european_call_option.get_price()
    put_price = european_put_option.get_price()

    # Verify put-call parity
    assert abs(call_price - put_price - 100 + 100 * np.exp(-0.05 * expiry)) < 0.01


def test_european_barrier_option() -> None:
    option = EuropeanPutOptionBarrierOut(
        r=0.05,
        sigma=0.20,
        S0=100,
        div=0.00,
        expiry=1,
        K=100,
        beta=0.5
    )

    option.price_approx(1000)

    print(option.get_price())


def test_american_option() -> None:
    option = EuropeanCallOption(
        T=10,
        r=0.20,
        S_0=100,
        sigma=0.20,
        K=100
    )

    price = option.price_approx(10000)

    assert abs(price - 86.4712) < 0.01