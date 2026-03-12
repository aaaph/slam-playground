from pathlib import Path
from typing import Any, Self, TypeVar, cast

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from yaml import safe_load

import gtsam
from core.transformations.special_euclidian_3_dim import SE3
from dataset.sensor_interfaces import (
    CameraConfigOptions,
    CameraConfigOptionsKeys,
    IMUConfigOptions,
    IMUConfigOptionsKeys,
    TransformMatrix,
)

T = TypeVar("T", bound=dict[str, Any])


class SensorConfig[T]:
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
        self._undistorted_k, self._undistorted_roi = cv2.getOptimalNewCameraMatrix(
            cameraMatrix=self.k,
            distCoeffs=self.distortion_coefficients,
            imageSize=self.resolution,
            alpha=0.0,
            newImgSize=self.resolution,
        )

    @property
    def resolution(self) -> tuple[int, int]:
        """Get the resolution of the camera."""
        return self.payload.get("resolution")

    @property
    def body_sensor_transform(self) -> np.ndarray:
        """Get the body->sensor transform."""
        t_bs: TransformMatrix = self.payload.get("T_BS")
        data: list[float] = t_bs.get("data")

        return np.array(data).reshape(t_bs.get("rows"), t_bs.get("cols"))

    @property
    def body_sensor_transform_rotation(self) -> Rotation:
        """Get the rotation of the body->sensor transform."""
        return Rotation.from_matrix(self.body_sensor_transform[:3, :3])

    @property
    def body_sensor_transform_translation(self) -> np.ndarray:
        """Get the translation of the body->sensor transform."""
        return self.body_sensor_transform[:3, 3]

    @property
    def intrinsics(self) -> tuple[float, float, float, float]:
        """Get the intrinsics of the camera."""
        return self.payload.get("intrinsics")

    @property
    def k(self) -> np.ndarray:
        """Get the camera matrix."""
        fx, fy, cx, cy = self.intrinsics
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    @property
    def k_undistorted(self) -> np.ndarray:
        """Get the undistorted camera matrix."""
        return self._undistorted_k

    @property
    def distortion_coefficients(self) -> np.ndarray:
        """Get the distortion coefficients of the camera."""
        return np.array(self.payload.get("distortion_coefficients"))

    def k_matrix_in_gtsam(self) -> gtsam.Cal3_S2:
        """Get the camera matrix in GTSAM format."""
        fx, fy, cx, cy = self.intrinsics
        skew = 0
        return gtsam.Cal3_S2(fx, fy, skew, cx, cy)

    @property
    def camera_in_body_se3(self) -> SE3:
        """Get the body->camera transform."""
        return SE3.from_matrix(self.body_sensor_transform)

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
    def body_sensor_transform(self) -> np.ndarray:
        """Get the body->sensor transform."""
        t_bs: TransformMatrix = self.payload.get("T_BS")
        data: list[float] = t_bs.get("data")

        return np.array(data).reshape(t_bs.get("rows"), t_bs.get("cols"))

    def __getitem__(
        self,
        key: IMUConfigOptionsKeys,
    ) -> Any:  # noqa: ANN401
        """Get an item from the camera configuration."""
        return self.payload[key]
