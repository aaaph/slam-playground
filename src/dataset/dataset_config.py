from pathlib import Path
from typing import Any, Self, TypeVar, cast

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from yaml import safe_load

from dataset.sensor_interfaces import (
    CameraConfigOptions,
    CameraConfigOptionsKeys,
    IMUConfigOptions,
    IMUConfigOptionsKeys,
    StereoConfigOptions,
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
    def distortion_coefficients(self) -> tuple[float, float, float, float]:
        """Get the distortion coefficients of the camera."""
        return self.payload.get("distortion_coefficients")

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


class StereoConfig(SensorConfig[StereoConfigOptions]):
    """Stereo configuration."""

    def __init__(self, cam0: CameraConfig, cam1: CameraConfig) -> None:
        """Initialize the stereo configuration."""
        super().__init__({})
        self.cam0 = cam0
        self.cam1 = cam1
        cam0_cam1_transform = np.linalg.inv(
            np.linalg.inv(self.cam0.body_sensor_transform) @ self.cam1.body_sensor_transform
        )
        r = Rotation.from_matrix(cam0_cam1_transform[:3, :3]).as_matrix()
        t = cam0_cam1_transform[:3, 3]

        # 7.0453063e-03 0.0070453063
        self.r = r
        self.t = t
        r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
            cameraMatrix1=np.array(self.cam0.k),
            distCoeffs1=np.array(self.cam0.distortion_coefficients),
            cameraMatrix2=np.array(self.cam1.k),
            distCoeffs2=np.array(self.cam1.distortion_coefficients),
            imageSize=self.cam0.resolution,
            R=np.array(r),
            T=np.array(t),
            flags=cv2.CALIB_ZERO_DISPARITY,
        )

        self.r1 = r1
        self.r2 = r2
        self.p1 = p1
        self.left_k_undistorted = p1[:3, :3].copy()
        self.p2 = p2
        self.q = q
        self.roi1 = roi1
        self.roi2 = roi2
        map1_x, map1_y = cv2.initUndistortRectifyMap(
            np.array(self.cam0.k),
            np.array(self.cam0.distortion_coefficients),
            r1,
            p1,
            self.cam0.resolution,
            cv2.CV_32FC1,
        )
        map2_x, map2_y = cv2.initUndistortRectifyMap(
            np.array(self.cam1.k),
            np.array(self.cam1.distortion_coefficients),
            r2,
            p2,
            self.cam1.resolution,
            cv2.CV_32FC1,
        )
        self.map1_x = map1_x
        self.map1_y = map1_y
        self.map2_x = map2_x
        self.map2_y = map2_y

    @property
    def left_map_x(self) -> np.ndarray:
        """Get the left map x."""
        return self.map1_x

    @property
    def left_map_y(self) -> np.ndarray:
        """Get the left map y."""
        return self.map1_y

    @property
    def right_map_x(self) -> np.ndarray:
        """Get the right map x."""
        return self.map2_x

    @property
    def right_map_y(self) -> np.ndarray:
        """Get the right map y."""
        return self.map2_y

    @property
    def k_rect_left(self) -> np.ndarray:
        """Get the left rectified camera matrix."""
        roi1 = self.roi1
        x, y, _, _ = roi1
        p1 = self.p1[:3, :3].copy()
        p1[0, 2] -= x
        p1[1, 2] -= y
        return p1

    @property
    def k_rect_right(self) -> np.ndarray:
        """Get the right rectified camera matrix."""
        roi2 = self.roi2
        x, y, _, _ = roi2
        p2 = self.p2[:3, :3].copy()
        p2[0, 2] -= x
        p2[1, 2] -= y
        return p2
