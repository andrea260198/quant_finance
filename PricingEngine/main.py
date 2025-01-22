from fastapi import FastAPI

from option_pricing.european_barrier_options import EuropeanPutOptionBarrierOut

app = FastAPI()


@app.get("/")
async def read_root():
    option = EuropeanPutOptionBarrierOut(
        r=0.05,
        sigma=0.20,
        S_0=100,
        div=0.00,
        T=1,
        strike=100,
        beta=0.5
    )

    price = option.price_approx(1000)

    return {"Hello": price}