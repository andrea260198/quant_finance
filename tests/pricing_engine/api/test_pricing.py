import unittest

from fastapi.testclient import TestClient

from pricing_engine.core.contract import PricingMethod
from pricing_engine.core.implied_volatility.heston_model import HestonEuropeanCallOption
from pricing_engine.core.interest_rates.short_rate_models import VasicekModel
from pricing_engine.core.interest_rates.zero_coupon_bond import ZeroCouponBond
from pricing_engine.core.option_pricing.american_options import AmericanCallOption
from pricing_engine.core.option_pricing.european_options import EuropeanCallOption
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
            T=10,
            r=0.20,
            S_0=100,
            sigma=0.20,
            strike=100
        )
        response = client.post(url, json=[option1.model_dump()])
        assert response.status_code == 200

        option2 = AmericanCallOption(
            T=10,
            r=0.20,
            S_0=100,
            sigma=0.20,
            strike=100
        )
        response = client.post(url, json=[option2.model_dump()])
        assert response.status_code == 200

        option3 = HestonEuropeanCallOption(
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

        short_rate_model = VasicekModel(
            dt=0.01,
            a=0.1,
            b=0.07,
            r0=0.02,
            sigma=0.02,
        )

        contracts = [
            EuropeanCallOption(
                T=10,
                r=0.20,
                S_0=100,
                sigma=0.20,
                strike=100
            ),
            AmericanCallOption(
                T=10,
                r=0.20,
                S_0=100,
                sigma=0.20,
                strike=100
            ),
            HestonEuropeanCallOption(
                T=10,
                r=0.20,
                S_0=100,
                sigma=0.20,
                strike=100
            ),
            ZeroCouponBond(
                T=10,
                short_rate_model=short_rate_model,
                pricing_method=PricingMethod.EXACT
            )
        ]
        response = client.post(url, json=[c.model_dump() for c in contracts])
        assert response.status_code == 200

    def test_delta(self):
        url = '/delta'

        contracts = [
            EuropeanCallOption(
                T=10,
                r=0.20,
                S_0=100,
                sigma=0.20,
                strike=100
            ),
            AmericanCallOption(
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
                T=10,
                r=0.20,
                S_0=100,
                sigma=0.20,
                strike=100
            ),
            AmericanCallOption(
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
                T=10,
                r=0.20,
                S_0=100,
                sigma=0.20,
                strike=100
            ),
            AmericanCallOption(
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
                T=10,
                r=0.20,
                S_0=100,
                sigma=0.20,
                strike=100
            ),
            AmericanCallOption(
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
                T=10,
                r=0.20,
                S_0=100,
                sigma=0.20,
                strike=100
            ),
            AmericanCallOption(
                T=10,
                r=0.20,
                S_0=100,
                sigma=0.20,
                strike=100
            ),
        ]
        response = client.post(url, json=[c.model_dump() for c in contracts])
        assert response.status_code == 200

