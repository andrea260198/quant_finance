from typing import Literal

from core.option_pricing.abstract_option import AbstractOption, ExactTrait
from core.contract import PricingMethod


class DeltaOne(AbstractOption, ExactTrait):
    pricing_method = PricingMethod.EXACT
    type: Literal["DeltaOne"] = "DeltaOne"

    def price_exact(self) -> float:
        raise NotImplementedError()
