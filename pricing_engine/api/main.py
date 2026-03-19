from typing import Union, Annotated

from fastapi import FastAPI
from pydantic import Field

from core.implied_volatility.heston_model import HestonEuropeanCallOption
from core.interest_rates.zero_coupon_bond import ZeroCouponBond
from core.option_pricing.american_options import AmericanCallOption, AmericanPutOption
from core.option_pricing.european_options import EuropeanCallOption, EuropeanPutOption

app = FastAPI()

annotated_contract_t = Annotated[
    Union[
        EuropeanCallOption,
        EuropeanPutOption,
        AmericanCallOption,
        AmericanPutOption,
        HestonEuropeanCallOption,
        ZeroCouponBond
    ],
    Field(discriminator="type")
]


@app.get("/helloworld")
async def test() -> dict[str, str]:
    return {"hello": "world"}


@app.post("/price")
async def price_contract(
        contract: annotated_contract_t
) -> float:
    return contract.price


@app.post("/price_multiple")
async def price_multiple_contracts(
        contracts: list[annotated_contract_t]
) -> list[float]:
    return [c.price for c in contracts]

