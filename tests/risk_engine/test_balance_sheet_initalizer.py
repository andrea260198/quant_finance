import unittest

import numpy as np
import polars as pl
from pydantic_extra_types.currency_code import ISO4217

from pricing_engine.core.interest_rates.zero_coupon_bond import ZeroCouponBond
from pricing_engine.core.option_pricing.european_options import (
    EuropeanCallOption,
    EuropeanPutOption,
)
from risk_engine.main import ScenaryCube, BalanceSheetInitalizer


class TestBalanceSheetInitalizer(unittest.TestCase):
    def test_simple_example(self):
        np.random.seed(0)

        scenary_cube = ScenaryCube(
            time=0,
            interest_rates=pl.DataFrame(
                data={
                    "currency": ["USD"] * 8,
                    "maturity": [0, 1, 2, 3, 4, 5, 6, 7],
                    "interest_rate": [0.05] * 8,
                }
            ),
            prices=pl.DataFrame(
                data={"underlying": ["AAPL", "GOOG", "MSFT"], "price": [150, 2800, 300]}
            ),
            volatilities=pl.DataFrame(
                data={
                    "underlying": ["AAPL", "GOOG", "MSFT"],
                    "volatility": [0.2, 0.25, 0.3],
                }
            ),
        )

        contracts = [
            EuropeanCallOption(
                strike=100,
                T=5,
                r=np.nan,
                S_0=np.nan,
                sigma=np.nan,
                currency=ISO4217("USD"),
                underlying_quantity=1,
                underlying_name="AAPL",
            ),
            EuropeanPutOption(
                strike=100,
                T=5,
                r=np.nan,
                S_0=np.nan,
                sigma=np.nan,
                currency=ISO4217("USD"),
                underlying_quantity=1,
                underlying_name="GOOG",
            ),
            ZeroCouponBond(
                T=5,
                short_rate_model=None,
                M=-1,  # Monte Carlo simulation sample size
                face_value=1000,
                currency=ISO4217("USD"),
            )
        ]
        initialized_contracts = BalanceSheetInitalizer(
            contracts=contracts, scenary_cube=scenary_cube
        ).run()

        portfolio_value = sum([contract.price for contract in initialized_contracts])

        assert portfolio_value == 852.4720681965857
