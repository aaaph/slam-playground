from dataclasses import dataclass

import cv2
import numpy as np

from core.camera_model.raw_stereo_config import RawStereoConfigDto
from core.camera_model.stereo_camera_ctx import StereoContext
from core.transformations.special_euclidian_3_dim import SE3
from dataset.dataset_config import CameraConfig


@dataclass
class StereoCameraModelConfig:
    """Stereo camera model configuration."""

    apply_equalization: bool = True
    apply_rectification: bool = True


class StereoCameraModel:
    """Stereo camera model."""

    def __init__(
        self, raw_config: RawStereoConfigDto, config: StereoCameraModelConfig, resolution: tuple[int, int]
    ) -> None:
        """Initialize the stereo camera model."""
        self.resolution = resolution
        self.config = config
        self.cam0 = raw_config.cam0
        self.cam1 = raw_config.cam1

        cam0_in_body = self.cam0.camera_in_body_se3
        cam1_in_body = self.cam1.camera_in_body_se3
        body_in_cam0 = cam0_in_body.inverse()
        cam0_in_cam1 = body_in_cam0 * cam1_in_body
        cam1_in_cam0 = cam0_in_cam1.inverse()

        cam1_in_cam0_rot = cam1_in_cam0.rotation().as_matrix()
        cam1_in_cam0_vec = cam1_in_cam0.translation()

        r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
            cameraMatrix1=self.cam0.k,
            distCoeffs1=self.cam0.distortion_coefficients,
            cameraMatrix2=self.cam1.k,
            distCoeffs2=self.cam1.distortion_coefficients,
            imageSize=self.cam0.resolution,
            R=np.array(cam1_in_cam0_rot),
            T=cam1_in_cam0_vec,
            flags=cv2.CALIB_ZERO_DISPARITY,
        )
        self.r1 = r1
        self.p1 = p1
        self.distortion_coefficients = self.cam0.distortion_coefficients
        map1_x, map1_y = cv2.initUndistortRectifyMap(
            self.cam0.k,
            self.cam0.distortion_coefficients,
            r1,
            p1,
            self.cam0.resolution,
            cv2.CV_32FC1,
        )
        map2_x, map2_y = cv2.initUndistortRectifyMap(
            self.cam1.k,
            self.cam1.distortion_coefficients,
            r2,
            p2,
            self.cam1.resolution,
            cv2.CV_32FC1,
        )
        self.map1_x = map1_x
        self.map1_y = map1_y
        self.map2_x = map2_x
        self.map2_y = map2_y

        self.stereo_k = p1[:3, :3].copy()
        self.baseline = float(-p2[0, 3] / self.stereo_k[0, 0])
        self.resolution = self.cam0.resolution

    @classmethod
    def from_cameras_config(cls, cam0: CameraConfig, cam1: CameraConfig) -> "StereoCameraModel":
        """Create a stereo camera model from a raw configuration."""
        resolution = cam0.resolution
        return cls(RawStereoConfigDto(cam0, cam1), StereoCameraModelConfig(), resolution)

    def _rectify_stereo(self, left_image: np.ndarray, right_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rectify the stereo images."""
        rect_left = cv2.remap(left_image, self.map1_x, self.map1_y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right_image, self.map2_x, self.map2_y, cv2.INTER_LINEAR)
        return rect_left, rect_right

    def _equalize_stereo(self, left_image: np.ndarray, right_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Equalize the stereo images."""
        equalized_left = cv2.equalizeHist(left_image)
        equalized_right = cv2.equalizeHist(right_image)
        return equalized_left, equalized_right

    def process_stereo(self, left_image: np.ndarray, right_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Process the stereo images."""
        if self.config.apply_equalization:
            left_image, right_image = self._equalize_stereo(left_image, right_image)
        if self.config.apply_rectification:
            left_image, right_image = self._rectify_stereo(left_image, right_image)
        return left_image, right_image

    def k_matricies(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the k matricies."""
        return (self.stereo_k, self.cam0.k, self.cam1.k)

    def body_sensor_transforms(self) -> tuple[SE3, SE3]:
        """Get the body sensor transforms."""
        return (self.cam0.camera_in_body_se3, self.cam1.camera_in_body_se3)

    def as_stereo_ctx(self) -> StereoContext:
        """Convert the stereo camera model to a stereo context."""
        return StereoContext(
            stereo_k=self.stereo_k,
            cam0_k=self.cam0.k,
            cam1_k=self.cam1.k,
            baseline=self.baseline,
            cam0_in_body_se3=self.cam0.camera_in_body_se3,
            cam1_in_body_se3=self.cam1.camera_in_body_se3,
            resolution=self.resolution,
        )
