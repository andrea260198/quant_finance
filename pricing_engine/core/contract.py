from enum import StrEnum
from functools import cached_property
from typing import Literal

from support.quant_dataclass import ImmutableDataclass


class PricingMethod(StrEnum):
    EXACT = "EXACT"
    BINOMIAL_TREE = "BINOMIAL_TREE"
    MONTE_CARLO = "MONTE_CARLO"
    QUASI_MONTE_CARLO = "QUASI_MONTE_CARLO"


class Contract(ImmutableDataclass):
    type: Literal["Contract"]
    pricing_method: PricingMethod

    @cached_property
    def price(self) -> float:
        raise NotImplementedError()



