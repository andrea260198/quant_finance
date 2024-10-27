from abc import ABC, abstractmethod
from dataclasses import dataclass

from overrides import override
from risk_management.model_results import ModelResults


@dataclass
class ModelParameters:
    scenarios_n: int


@dataclass
class RiskModel(ABC):
    model_params: ModelParameters

    @abstractmethod
    def run(self):
        pass


@dataclass
class MyRiskModel(RiskModel):
    @override
    def run(self) -> ModelResults:
        return ModelResults()




