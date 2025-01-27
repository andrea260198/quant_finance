import requests

from implied_volatility.heston_model import HestonEuropeanCallOption
from option_pricing.american_options import AmericanCallOption
from option_pricing.european_options import EuropeanCallOption


if __name__ == '__main__':
    url = 'http://localhost:8000/contract/'

    data = EuropeanCallOption(
        T=10,
        r=0.20,
        S_0=100,
        sigma=0.20,
        strike=100
    ).dict()
    response = requests.post(url, json=data)
    print(response.json())

    data = AmericanCallOption(
        T=10,
        r=0.20,
        S_0=100,
        sigma=0.20,
        strike=100
    ).dict()
    response = requests.post(url, json=data)
    print(response.json())

    data = HestonEuropeanCallOption(
        T=10,
        r=0.20,
        S_0=100,
        sigma=0.20,
        strike=100
    ).dict()
    response = requests.post(url, json=data)
    print(response.json())