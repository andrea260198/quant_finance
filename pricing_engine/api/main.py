from typing import Union, Annotated

from fastapi import FastAPI
from pydantic import Field

from core.implied_volatility.heston_model import HestonEuropeanCallOption
from core.option_pricing.american_options import AmericanCallOption
from core.option_pricing.european_options import EuropeanCallOption

app = FastAPI()


@app.get("/helloworld")
async def test() -> dict[str, str]:
    return {"hello": "world"}


@app.post("/price")
async def price_contract(contract: Annotated[Union[EuropeanCallOption, AmericanCallOption, HestonEuropeanCallOption], Field(discriminator="type")]) -> float:
    return contract.price

