from implied_volatility.heston_model import HestonEuropeanCallOption, plot_volatility_surface
from option_pricing.american_options import AmericanCallOption
from option_pricing.european_options import EuropeanCallOption

if __name__ == '__main__':
    plot_volatility_surface()

    quit()

    option = HestonEuropeanCallOption(
        T=1,
        r=0.20,
        S_0=100,
        sigma=0.20,
        K=100
    )

    price = option.price_mc_approx(10_000_000, 0.01)

    print(price)