from pydantic import BaseModel, ConfigDict


class ImmutableDataclass(BaseModel):
    """
    Dataclass base model which enforces type correctness at runtime

    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True
    )