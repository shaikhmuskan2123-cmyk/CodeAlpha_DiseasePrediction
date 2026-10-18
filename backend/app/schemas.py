from typing import Any

from pydantic import BaseModel, ConfigDict


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any]
    persist: bool | None = None