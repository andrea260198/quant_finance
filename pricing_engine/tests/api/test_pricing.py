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
    def test_price_single_constract(self):
        url = '/price'

        option1 = EuropeanCallOption(
            type="EuropeanCallOption",
            T=10,
            r=0.20,
            S_0=100,
            sigma=0.20,
            strike=100
        )
        response = client.post(url, json=[option1.model_dump()])
        assert response.status_code == 200

        option2 = AmericanCallOption(
            type="AmericanCallOption",
            T=10,
            r=0.20,
            S_0=100,
            sigma=0.20,
            strike=100
        )
        response = client.post(url, json=[option2.model_dump()])
        assert response.status_code == 200

        option3 = HestonEuropeanCallOption(
            type="HestonEuropeanCallOption",
            T=10,
            r=0.20,
            S_0=100,
            sigma=0.20,
            strike=100
        )
        response = client.post(url, json=[option3.model_dump()])
        assert response.status_code == 200

    def test_price_multiple_contracts(self):
        url = '/price'

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

    def test_delta(self):
        url = '/delta'

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
        ]
        response = client.post(url, json=[c.model_dump() for c in contracts])
        assert response.status_code == 200

    def test_rho(self):
        url = '/rho'

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
        ]
        response = client.post(url, json=[c.model_dump() for c in contracts])
        assert response.status_code == 200

    def test_theta(self):
        url = '/theta'

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
        ]
        response = client.post(url, json=[c.model_dump() for c in contracts])
        assert response.status_code == 200

    def test_vega(self):
        url = '/vega'

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
        ]
        response = client.post(url, json=[c.model_dump() for c in contracts])
        assert response.status_code == 200

    def test_gamma(self):
        url = '/gamma'

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
        ]
        response = client.post(url, json=[c.model_dump() for c in contracts])
        assert response.status_code == 200

