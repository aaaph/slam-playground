from pathlib import Path
from typing import Any, Generic, Self, TypeVar, cast

import jax as x
from scipy.spatial.transform import Rotation
from yaml import safe_load

from dataset.sensor_interfaces import (
    CameraConfigOptions,
    CameraConfigOptionsKeys,
    IMUConfigOptions,
    IMUConfigOptionsKeys,
    TransformMatrix,
)

T = TypeVar("T", bound=dict[str, Any])


class SensorConfig(Generic[T]):
    """Generic sensor configuration that can work with any sensor type."""

    def __init__(self, payload: T) -> None:
        """Initialize the sensor configuration."""
        self.payload = payload

    @classmethod
    def from_yaml(cls: type[Self], file_path: str) -> Self:
        """
        Create a sensor configuration from a file.

        Args:
            file_path: Path to the YAML configuration file
            model: Optional model class to instantiate the payload with.
                  If None, the raw dictionary will be used as payload.

        """
        file = Path(file_path)
        if not file.exists():
            raise FileNotFoundError("Config file does not exist")

        with Path.open(file_path, "r") as file:
            raw_payload = safe_load(file)
            payload: dict[str, Any] = cast("dict[str, Any]", raw_payload)
            return cls(payload)

    def __getitem__(self, key: str) -> T:
        """Get an item from the configuration."""
        return self.payload[key]

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the configuration."""
        return key in self.payload


class CameraConfig(SensorConfig[CameraConfigOptions]):
    """Camera configuration."""

    def __init__(self, payload: CameraConfigOptions) -> None:
        """Initialize the camera configuration."""
        super().__init__(payload)

    @property
    def resolution(self) -> tuple[int, int]:
        """Get the resolution of the camera."""
        return self.payload.get("resolution")

    @property
    def body_sensor_transform(self) -> x.Array:
        """Get the body->sensor transform."""
        t_bs: TransformMatrix = self.payload.get("T_BS")
        data: list[float] = t_bs.get("data")

        return x.numpy.array(data).reshape(t_bs.get("rows"), t_bs.get("cols"))

    @property
    def body_sensor_transform_rotation(self) -> Rotation:
        """Get the rotation of the body->sensor transform."""
        return Rotation.from_matrix(self.body_sensor_transform[:3, :3])

    @property
    def body_sensor_transform_translation(self) -> x.Array:
        """Get the translation of the body->sensor transform."""
        return self.body_sensor_transform[:3, 3]

    def __getitem__(
        self,
        key: CameraConfigOptionsKeys,
    ) -> Any:  # noqa: ANN401
        """Get an item from the camera configuration."""
        return self.payload[key]


class IMUConfig(SensorConfig[IMUConfigOptions]):
    """IMU configuration."""

    def __init__(self, payload: IMUConfigOptions) -> None:
        """Initialize the IMU configuration."""
        super().__init__(payload)

    @property
    def body_sensor_transform(self) -> x.Array:
        """Get the body->sensor transform."""
        t_bs: TransformMatrix = self.payload.get("T_BS")
        data: list[float] = t_bs.get("data")

        return x.numpy.array(data).reshape(t_bs.get("rows"), t_bs.get("cols"))

    def __getitem__(
        self,
        key: IMUConfigOptionsKeys,
    ) -> Any:  # noqa: ANN401
        """Get an item from the camera configuration."""
        return self.payload[key]
