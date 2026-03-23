from core.implied_volatility.heston_model import HestonEuropeanCallOption


def test_heston_european_call_option() -> None:
    option = HestonEuropeanCallOption(
        T=1,
        r=0.20,
        S_0=100,
        sigma=0.20,
        strike=100
    )

    price = option.price_mc_approx(1_000_000, 0.01)