from abc import ABC, abstractmethod
from overrides import override
from risk_management.model_results import ModelResults
from support.quant_dataclass import QuantDataclass


class ModelParameters(QuantDataclass):
    scenarios_n: int


class RiskModel(QuantDataclass, ABC):
    model_params: ModelParameters

    @abstractmethod
    def run(self) -> ModelResults:
        pass


class MyRiskModel(RiskModel):
    @override
    def run(self) -> ModelResults:
        return ModelResults()




