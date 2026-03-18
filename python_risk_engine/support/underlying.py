from functools import cached_property

from support.quant_dataclass import QuantDataclass


class Underlying(QuantDataclass):
    _price: float

    @cached_property
    def price(self):
        return self._price