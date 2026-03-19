import unittest

from fastapi.testclient import TestClient

from core.implied_volatility.heston_model import HestonEuropeanCallOption
from core.option_pricing.american_options import AmericanCallOption
from core.option_pricing.european_options import EuropeanCallOption
from api.main import app

client = TestClient(app)


class TestHelloWorld(unittest.TestCase):
    def test_hello_world(self):
        url = '/helloworld'
        response = client.get(url)
        assert response.status_code == 200
        assert  response.json() == {"hello": "world"}


class TestOption(unittest.TestCase):
    def test_price(self):
        url = '/price'

        data = EuropeanCallOption(
            type="EuropeanCallOption",
            T=10,
            r=0.20,
            S_0=100,
            sigma=0.20,
            strike=100
        ).model_dump()
        response = client.post(url, json=data)
        assert response.status_code == 200

        data = AmericanCallOption(
            type="AmericanCallOption",
            T=10,
            r=0.20,
            S_0=100,
            sigma=0.20,
            strike=100
        ).model_dump()
        response = client.post(url, json=data)
        assert response.status_code == 200

        data = HestonEuropeanCallOption(
            type="HestonEuropeanCallOption",
            T=10,
            r=0.20,
            S_0=100,
            sigma=0.20,
            strike=100
        ).model_dump()
        response = client.post(url, json=data)
        assert response.status_code == 200

    def test_price_multiple(self):
        url = '/price_multiple'

        contracts = [
            EuropeanCallOption(
                type="EuropeanCallOption",
                T=10,
                r=0.20,
                S_0=100,
                sigma=0.20,
                strike=100
            ),
                AmericanCallOption(
                type="AmericanCallOption",
                T=10,
                r=0.20,
                S_0=100,
                sigma=0.20,
                strike=100
            ),
                HestonEuropeanCallOption(
                type="HestonEuropeanCallOption",
                T=10,
                r=0.20,
                S_0=100,
                sigma=0.20,
                strike=100
            )
        ]
        response = client.post(url, json=[c.model_dump() for c in contracts])
        assert response.status_code == 200

