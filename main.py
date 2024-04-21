from implied_volatility.heston_model import HestonEuropeanCallOption
from option_pricing.american_options import AmericanCallOption
from option_pricing.european_options import EuropeanCallOption

if __name__ == '__main__':

    option = HestonEuropeanCallOption(
        T=1,
        r=0.20,
        S_0=100,
        sigma=0.20,
        K=100
    )

    price = option.price_mc_approx(10000, 0.01)

    print(price)