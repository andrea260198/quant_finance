from typing import Literal

from core.option_pricing.abstract_option import AbstractOption, ExactTrait, PricingMethod


class DeltaOne(AbstractOption, ExactTrait):
    type: Literal["DeltaOne"]
    pricing_method = PricingMethod.EXACT
    _price: float = 0.0

    def price_exact(self) -> float:
        return self._price
