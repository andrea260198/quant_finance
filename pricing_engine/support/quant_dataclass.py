from pydantic import BaseModel, ConfigDict


class ImmutableDataclass(BaseModel):
    """
    Dataclass which enforces type correctness and immutability at runtime
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True
    )