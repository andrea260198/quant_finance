from pydantic import BaseModel, ConfigDict


class QuantDataclass(BaseModel):
    """
    Dataclass base model which enforces type correctness at runtime

    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True
    )