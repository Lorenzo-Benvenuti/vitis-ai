import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Backbone = Literal[
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "mobilenet_v3_small",
]

# variety becomes a directory name, so it is restricted to safe characters.
# This is what blocks path traversal.
_VARIETY_PATTERN = re.compile(r"[\w -]+")


class UploadMetadata(BaseModel):
    variety: str

    @field_validator("variety")
    @classmethod
    def _validate_variety(cls, value: str) -> str:
        value = value.strip()
        if not _VARIETY_PATTERN.fullmatch(value):
            raise ValueError("variety contains invalid characters")
        return value


class UploadResponse(BaseModel):
    uploaded: int
    train_count: int
    validation_count: int
    variety: str


class AugmentationConfig(BaseModel):
    horizontal_flip: bool = True
    vertical_flip_p: float = 0.2
    rotation: float = 30.0
    color_jitter_brightness: float = 0.4
    color_jitter_contrast: float = 0.4
    color_jitter_saturation: float = 0.5
    color_jitter_hue: float = 0.1
    random_erasing_p: float = 0.3


class ModelTrainRequest(BaseModel):
    epochs: int = Field(default=20, gt=0)
    lr: float = Field(default=0.001, gt=0)
    batch_size: int = Field(default=32, gt=0)
    backbone: Backbone = "resnet18"
    augmentation: AugmentationConfig = AugmentationConfig()


class ModelTrainResponse(BaseModel):
    status: str = "started"
    message: str = "Training started in the background."


class PredictResponse(BaseModel):
    variety: str
    # Softmax probability of the predicted class. Bounding it here means the
    # schema documents and enforces the invariant rather than assuming it.
    confidence: float = Field(ge=0.0, le=1.0)
