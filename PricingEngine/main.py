from typing import Union

from fastapi import FastAPI

from option_pricing.abstract_option import PricingMethod, AbstractOption
from option_pricing.american_options import AmericanCallOption
from option_pricing.european_options import EuropeanCallOption

app = FastAPI()


@app.get("/")
async def test():
    return {"Hello": "World"}


@app.post("/contract/")
async def price_contract(contract: Union[AmericanCallOption, EuropeanCallOption]):
    print(contract.type)
    match contract.type:
        case "EuropeanCallOption" as name:
            option = EuropeanCallOption.parse_raw(contract.json())
            return {name: option.price}
        case _:
            return {"Error": contract.type}

