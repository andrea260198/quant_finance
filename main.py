from implied_volatility.heston_model import HestonEuropeanCallOption


if __name__ == '__main__':

    option = HestonEuropeanCallOption(
        T=1,
        r=0.20,
        S_0=100,
        sigma=0.20,
        strike=100
    )

    price = option.price_mc_approx(10_000_000, 0.01)

    print(price)