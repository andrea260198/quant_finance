from typing import Union

from fastapi import FastAPI

from core.implied_volatility.heston_model import HestonEuropeanCallOption
from core.option_pricing.american_options import AmericanCallOption
from core.option_pricing.european_options import EuropeanCallOption

app = FastAPI()


@app.get("/")
async def test():
    return {"Hello": "World"}


@app.post("/contract/")
async def price_contract(contract: Union[EuropeanCallOption, AmericanCallOption, HestonEuropeanCallOption]) -> dict[str, float]:
    print(type(contract))
    print(contract.type)
    match contract.type:
        case "EuropeanCallOption" as name:
            option = EuropeanCallOption.parse_raw(contract.json())
            return {name: option.price}
        case "AmericanCallOption" as name:
            option = AmericanCallOption.parse_raw(contract.json())
            return {name: option.price_approx(500)}
        case "HestonEuropeanCallOption" as name:
            option = HestonEuropeanCallOption.parse_raw(contract.json())
            return {name: option.price}
        case _:
            return {f"Error for {contract.type}": 0.0}

