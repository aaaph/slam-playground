from __future__ import annotations

from pathlib import Path  # noqa: TC003 - pydantic needs Path in the runtime model namespace.

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TransformMatrixConfig(BaseModel):
    """Homogeneous transform matrix stored in dataset rig YAML."""

    rows: int
    cols: int
    data: list[float]

    @model_validator(mode="after")
    def validate_shape(self) -> TransformMatrixConfig:
        """Validate that flat matrix data matches rows and columns."""
        expected_items = self.rows * self.cols
        if len(self.data) != expected_items:
            msg = f"Transform matrix has {len(self.data)} values, expected {expected_items}"
            raise ValueError(msg)
        return self


class CameraRigConfig(BaseModel):
    """Camera calibration in normalized rig config."""

    model_config = ConfigDict(populate_by_name=True)

    rate_hz: float
    resolution: tuple[int, int]
    camera_model: str
    intrinsics: tuple[float, float, float, float]
    distortion_model: str
    distortion_coefficients: list[float]
    body_sensor_transform: TransformMatrixConfig = Field(alias="T_BS")


class ImuRigConfig(BaseModel):
    """IMU calibration and noise parameters in normalized rig config."""

    model_config = ConfigDict(populate_by_name=True)

    rate_hz: float
    body_sensor_transform: TransformMatrixConfig = Field(alias="T_BS")
    gyroscope_noise_density: float
    gyroscope_random_walk: float
    accelerometer_noise_density: float
    accelerometer_random_walk: float


class DatasetRigConfig(BaseModel):
    """Normalized sensor rig config for a supported dataset family."""

    name: str
    cam0: CameraRigConfig
    cam1: CameraRigConfig
    imu0: ImuRigConfig


class DatasetStreamsConfig(BaseModel):
    """Dataset stream file layout relative to dataset root."""

    cam0: Path
    cam1: Path
    imu0: Path
    ground_truth: Path | None = None


class DatasetManifest(BaseModel):
    """Dataset manifest resolved by name."""

    name: str
    type: str
    root: Path
    rig: Path
    streams: DatasetStreamsConfig
    cache: Path | None = None


class ResolvedDatasetManifest(BaseModel):
    """Dataset manifest resolved together with its sensor rig."""

    dataset: DatasetManifest
    rig: DatasetRigConfig
