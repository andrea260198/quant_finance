from fastapi.testclient import TestClient

from core.implied_volatility.heston_model import HestonEuropeanCallOption
from core.option_pricing.american_options import AmericanCallOption
from core.option_pricing.european_options import EuropeanCallOption
from api.main import app

client = TestClient(app)


def test_server():
    url = '/price'

    data = EuropeanCallOption(
        type="EuropeanCallOption",
        T=10,
        r=0.20,
        S_0=100,
        sigma=0.20,
        strike=100
    ).dict()
    response = client.post(url, json=data)
    print(response.json())

    data = AmericanCallOption(
        type="AmericanCallOption",
        T=10,
        r=0.20,
        S_0=100,
        sigma=0.20,
        strike=100
    ).dict()
    response = client.post(url, json=data)
    print(response.json())

    data = HestonEuropeanCallOption(
        type="HestonEuropeanCallOption",
        T=10,
        r=0.20,
        S_0=100,
        sigma=0.20,
        strike=100
    ).dict()
    response = client.post(url, json=data)
    print(response.json())
