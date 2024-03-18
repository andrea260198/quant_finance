from option_pricing.american_options import AmericanCallOption
from option_pricing.european_options import EuropeanCallOption

if __name__ == '__main__':

    option1 = AmericanCallOption(
        T = 10,
        r = 0.20,
        S_0 = 100,
        sigma = 0.20,
        K = 100
    )

    price = option1.price_approx(10000)


    option2 = EuropeanCallOption(
        T=10,
        r=0.20,
        S_0=100,
        sigma=0.20,
        K=100
    )

    price = option2.price_approx(10000)

    pass