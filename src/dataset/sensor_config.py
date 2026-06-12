from pathlib import Path
from typing import Any, Self, cast

import cv2
import gtsam
import numpy as np
from scipy.spatial.transform import Rotation
from yaml import safe_load

from core.transformations.special_euclidian_3_dim import SE3
from dataset.manifest import CameraRigConfig, ImuRigConfig
from dataset.sensor_interfaces import (
    CameraConfigOptions,
    IMUConfigOptions,
    TransformMatrix,
)


class Sensor:
    """Runtime wrapper around a parsed sensor calibration payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Initialize sensor."""
        self.payload = payload

    @classmethod
    def from_yaml(cls: type[Self], file_path: str | Path) -> Self:
        """Create a sensor configuration from a YAML file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError("Config file does not exist")

        with path.open("r", encoding="utf-8") as f:
            raw_payload = safe_load(f)
            payload: dict[str, Any] = cast("dict[str, Any]", raw_payload)
            return cls(payload)

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Get an item from the configuration."""
        return self.payload[key]

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the configuration."""
        return key in self.payload

    def _transform_matrix(self, key: str = "T_BS") -> np.ndarray:
        transform: TransformMatrix = cast("TransformMatrix", self.payload.get(key, {}))
        data: list[float] = transform.get("data", [])
        return np.array(data).reshape(transform.get("rows", 0), transform.get("cols", 0))


class CameraSensor(Sensor):
    """Runtime camera calibration."""

    def __init__(self, payload: CameraConfigOptions) -> None:
        """Initialize camera sensor."""
        super().__init__(cast("dict[str, Any]", payload))
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
        return self.payload.get("resolution", (0, 0))

    @property
    def body_sensor_transform(self) -> np.ndarray:
        """Get the body->sensor transform."""
        return self._transform_matrix()

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
        return self.payload.get("intrinsics", (0, 0, 0, 0))

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

    @classmethod
    def from_rig_config(cls, cam_rig_config: CameraRigConfig) -> Self:
        """Create a camera sensor from a rig configuration."""
        return cls(cam_rig_config.model_dump(by_alias=True, exclude={"rate_hz"}))


class IMUSensor(Sensor):
    """Runtime IMU calibration."""

    def __init__(self, payload: IMUConfigOptions) -> None:
        """Initialize IMU sensor."""
        super().__init__(cast("dict[str, Any]", payload))

    @property
    def body_sensor_transform(self) -> np.ndarray:
        """Get the body->sensor transform."""
        return self._transform_matrix()

    @classmethod
    def from_rig_config(cls, imu_rig_config: ImuRigConfig) -> Self:
        """Create an IMU sensor from a rig configuration."""
        return cls(imu_rig_config.model_dump(by_alias=True))
