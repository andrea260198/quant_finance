from abc import ABC, abstractmethod
from overrides import override
from model_results import ModelResults


class RiskModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass


class MyRiskModel(RiskModel):
    @override
    def __init__(self):
        pass

    @override
    def run(self) -> ModelResults:
        return ModelResults()




