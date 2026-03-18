import requests

from core.implied_volatility.heston_model import HestonEuropeanCallOption
from core.option_pricing.american_options import AmericanCallOption
from core.option_pricing.european_options import EuropeanCallOption


def main():
    url = 'http://localhost:8000/contract/'

    data = EuropeanCallOption(
        T=10,
        r=0.20,
        S_0=100,
        sigma=0.20,
        strike=100
    ).dict()
    response = requests.get(url, json=data)
    print(response.json())

    data = AmericanCallOption(
        T=10,
        r=0.20,
        S_0=100,
        sigma=0.20,
        strike=100
    ).dict()
    response = requests.get(url, json=data)
    print(response.json())

    data = HestonEuropeanCallOption(
        T=10,
        r=0.20,
        S_0=100,
        sigma=0.20,
        strike=100
    ).dict()
    response = requests.get(url, json=data)
    print(response.json())


if __name__ == '__main__':
    main()