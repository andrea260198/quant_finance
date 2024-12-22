from pydantic import BaseModel, ConfigDict


class QuantDataclass(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True
    )