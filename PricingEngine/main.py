from fastapi import FastAPI

from option_pricing.abstract_option import PricingMethod
from option_pricing.european_options import EuropeanCallOption

app = FastAPI()


@app.get("/")
async def test():
    return {"Hello": "World"}


@app.get("/price/{contract_name}")
async def price_contract(contract_name: str):
    print(contract_name)
    match contract_name:
        case "EuropeanCallOption":
            option = EuropeanCallOption(
                T=10,
                r=0.05,
                S_0=100,
                sigma=0.20,
                strike=100,
                pricing_method=PricingMethod.EXACT
            )
            return {contract_name: option.price}
        case _:
            return {"Error": contract_name}

