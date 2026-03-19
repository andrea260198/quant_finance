from functools import cached_property

from support.quant_dataclass import ImmutableDataclass


class Underlying(ImmutableDataclass):
    _price: float

    @cached_property
    def price(self):
        return self._price