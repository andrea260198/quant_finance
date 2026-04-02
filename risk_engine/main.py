import polars as pl

from pricing_engine.core.interest_rates.short_rate_models import CoxIngersolRossModel
from pricing_engine.core.interest_rates.zero_coupon_bond import ZeroCouponBond
from pricing_engine.core.option_pricing.european_options import EuropeanCallOption, EuropeanPutOption
from pricing_engine.support.quant_dataclass import ImmutableDataclass
from pricing_engine.core.contract import Contract, PricingMethod
import numpy as np


class ScenaryCube(ImmutableDataclass):
    time: int
    interest_rates: pl.DataFrame
    prices: pl.DataFrame
    volatilities: pl.DataFrame

    def get_interest_rate(
        self,
        currency,
        maturity,
    ) -> float:
        return self.interest_rates.filter(
            (pl.col("currency") == currency),
            (pl.col("maturity") == maturity),
        )["interest_rate"].item()


    def get_underlying_price(
        self,
        underlying_name: str
    ) -> float:
        return self.prices.filter(
            pl.col("underlying") == underlying_name
        )["price"].item()

    def get_underlying_volatility(
        self,
        underlying
    ) -> float:
        return self.volatilities.filter(
            pl.col("underlying") == underlying
        )["volatility"].item()




class ScenarioGenerator(ImmutableDataclass):
    starting_conditions: pl.DataFrame

    def run(self) -> ScenaryCube:
        """
        Generate final scenario adding random component and drift component to `starting_conditions`
        """
        final_scenario = pl.DataFrame()
        return final_scenario



class BalanceSheetInitalizer(ImmutableDataclass):
    contracts: list[Contract]
    scenary_cube: ScenaryCube

    def run(self) -> list[Contract]:
        initialized_contracts = [self.initialize_contract(contract) for contract in self.contracts]
        return initialized_contracts

    def initialize_contract(self, contract: Contract) -> Contract:
        match contract:
            case EuropeanCallOption() as european_call_option:
                r = self.scenary_cube.get_interest_rate(
                    currency=european_call_option.currency,
                    maturity=european_call_option.T - self.scenary_cube.time,
                )
                S_0 = self.scenary_cube.get_underlying_price(
                    underlying_name=european_call_option.underlying_name
                )
                sigma = self.scenary_cube.get_underlying_volatility(
                    underlying=european_call_option.underlying_name
                )

                return EuropeanCallOption(
                    T=european_call_option.T - self.scenary_cube.time,
                    r=r,
                    S_0=S_0,
                    sigma=sigma,
                    strike=european_call_option.strike,
                    underlying_name=european_call_option.underlying_name,
                    currency=european_call_option.currency
                )
            case EuropeanPutOption() as european_put_option:
                r = self.scenary_cube.get_interest_rate(
                    currency=european_put_option.currency,
                    maturity=european_put_option.T - self.scenary_cube.time,
                )
                S_0 = self.scenary_cube.get_underlying_price(
                    underlying_name=european_put_option.underlying_name
                )
                sigma = self.scenary_cube.get_underlying_volatility(
                    underlying=european_put_option.underlying_name
                )

                return EuropeanPutOption(
                    T=european_put_option.T - self.scenary_cube.time,
                    r=r,
                    S_0=S_0,
                    sigma=sigma,
                    strike=european_put_option.strike,
                    underlying_name=european_put_option.underlying_name,
                    currency=european_put_option.currency
                )
            case ZeroCouponBond() as zero_coupon_bond:
                return ZeroCouponBond(
                    T=zero_coupon_bond.T - self.scenary_cube.time,
                    short_rate_model=CoxIngersolRossModel(
                        dt=0.01,
                        a=0.1,
                        b=0.05,
                        r0=self.scenary_cube.get_interest_rate(
                            currency=zero_coupon_bond.currency,
                            maturity=0,
                        ),
                        sigma=0.01
                    ),
                    M=100_000,
                    face_value=zero_coupon_bond.face_value,
                    currency=zero_coupon_bond.currency,
                )
            case _:
                raise ValueError(f"Contract type {contract.type} not supported")


class BalanceSheet(ImmutableDataclass):
    assets: list[Contract]
    liabilities: list[Contract]
    scenary_cube: ScenaryCube

    def compute_equity(self) -> float:
        """
        Combine current scenario and items to obtain the
        """
        scenary_cube = ScenaryCube(
            time = 0,
            interest_rates = pl.DataFrame(data={"currency": ["USD"] * 8, "maturity": [0, 1, 2, 3, 4, 5, 6, 7], "interest_rate": [0.05] * 8}),
            prices = pl.DataFrame(data={"underlying": ["AAPL", "GOOG", "MSFT"], "price": [150, 2800, 300]}),
            volatilities = pl.DataFrame(data={"underlying": ["AAPL", "GOOG", "MSFT"], "volatility": [0.2, 0.25, 0.3]}),
        )

        contracts = [EuropeanCallOption(strike=100, T=10, r=np.nan, S_0=np.nan, sigma=np.nan, currency="USD", underlying_quantity=1)]
        initialized_contracts = self.initialize_contracts(contracts, scenary_cube)

        portfolio_value = sum([contract.price for contract in initialized_contracts])

        return portfolio_value


        assets_total = sum([asset.price for asset in BalanceSheetInitalizer(self.assets, self.scenario).run()])
        liabilites_total = sum([liability.price for liability in BalanceSheetInitalizer(self.liabilities, self.scenario).run()])
        equity = assets_total - liabilites_total
        return equity


def main():
    #world = pl.DataFrame({"scenario_id": [0], ""})
    assets = pl.DataFrame(data={"contract_type": ["EuropeanCallOption", "EuropeanPutOption"], "params_json": ["", ""]})


if __name__ == "__main__":
    main()
