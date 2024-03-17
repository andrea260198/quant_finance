from abc import ABC, abstractmethod


class AbstractOption(ABC):
    def __init__(self) -> None:
        self._price: float | None = None

    @abstractmethod
    def price_approx(self, N: int) -> float:
        pass

    def get_price(self) -> float:
        if self._price is not None:
            return self._price
        else:
            raise Exception("Price not yet computed.")