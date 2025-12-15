from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from core.feature_tracker.feature import Feature
from core.transformations.helpers import skew
from core.transformations.special_euclidian_3_dim import SE3
from core.types.stereo_camera_dto import StereoCameraDto
from dataset.euroc import EurocConfig
from logger import spawn_logger


class FeatureTriangulationThresholds(NamedTuple):
    """Thresholds for feature triangulation."""

    depth_max_threshold: float = 40.0
    depth_min_threshold: float = 0.15
    disparity_min_threshold: float = 5.0
    vertical_shift_threshold: float = 10.0
    max_condition_number_threshold: float = 10000.0


_DEFAULT_THRESHOLDS = FeatureTriangulationThresholds()

StereoCameraMatrix = NDArray[np.float64]  # shape: (3, 3)
LeftCameraMatrix = NDArray[np.float64]  # shape: (3, 3)
RightCameraMatrix = NDArray[np.float64]  # shape: (3, 3)
CameraMatricies = tuple[StereoCameraMatrix, LeftCameraMatrix, RightCameraMatrix]
Baseline = float
BodySensorTransform = SE3  # shape: (4, 4)
LeftCameraInBodyTransform = BodySensorTransform
RightCameraInBodyTransform = BodySensorTransform
BodySensorTransforms = tuple[LeftCameraInBodyTransform, RightCameraInBodyTransform]
Matrix = NDArray[np.float64]  # shape: (3, 3)
Vector = NDArray[np.float64]  # shape: (3,)


class FeatureTriangulation:
    """Feature triangulation module."""

    def __init__(
        self,
        k_matricies: CameraMatricies,
        baseline: Baseline,
        body_sensor_transforms: BodySensorTransforms,
        thresholds: FeatureTriangulationThresholds = _DEFAULT_THRESHOLDS,
    ) -> None:
        """Initialize the feature triangulation module."""
        self.logger = spawn_logger(app="feature_triangulation")
        self.k_stereo, self.k_left, self.k_right = k_matricies
        self.baseline = baseline
        self.thresholds = thresholds
        self.cam0_in_body, self.cam1_in_body = body_sensor_transforms
        self.k_stereo_inv = np.linalg.inv(self.k_stereo)
        self.k_left_inv = np.linalg.inv(self.k_left)
        self.k_right_inv = np.linalg.inv(self.k_right)
        self.body_in_cam0 = self.cam0_in_body.inverse()
        self.body_in_cam1 = self.cam1_in_body.inverse()

    def make_initial_guess_by_stereo_pair(self, feature: Feature) -> tuple[bool, Vector]:
        """
        Make an initial guess for a feature over last stereo pair.

        Args:
            feature: The feature to make an initial guess for.

        Returns:
            GoodFeature: True if the feature is good, False otherwise.
            np.ndarray: The initial guess for the feature in camera frame.
            Shape: (3,)

        """
        active_stereo_pair = feature.get_active_stereo_pair()
        if active_stereo_pair is None:
            raise ValueError("Feature has no active stereo pair")
        _, left_uv, right_uv = active_stereo_pair
        if right_uv is None:
            raise ValueError("Feature has no active right point")
        left_u, left_v = left_uv
        right_u, right_v = right_uv
        fx = self.k_stereo[0, 0]
        fy = self.k_stereo[1, 1]
        cx = self.k_stereo[0, 2]
        cy = self.k_stereo[1, 2]
        baseline = self.baseline
        disp = left_u - right_u
        if disp <= 0:
            raise ValueError("Disparity is non-positive")

        z = fx * baseline / disp
        x = (left_u - cx) * z / fx
        y = (left_v - cy) * z / fy

        bad_parallax = disp < self.thresholds.disparity_min_threshold
        too_close = z < self.thresholds.depth_min_threshold
        too_far = z > self.thresholds.depth_max_threshold
        vertical_shift = abs(left_v - right_v) > self.thresholds.vertical_shift_threshold

        bad_feature = bad_parallax or too_close or too_far or vertical_shift

        return not bad_feature, np.array([x, y, z])

    def make_linear_triangulation_guess(self, feature: Feature, camera_in_world_se3: SE3) -> tuple[bool, Vector]:
        """Make a linear triangulation guess for a feature."""
        if not feature.ready_to_triangulate:
            return False, np.full(3, np.nan)
        try:
            solution, _, rank, s = np.linalg.lstsq(feature.A, feature.b, rcond=None)
            if rank < 3:  # noqa: PLR2004
                return False, np.full(3, np.nan)
            feat_in_world_vec = solution
        except np.linalg.LinAlgError:
            return False, np.full(3, np.nan)

        cond_a = np.inf if s[-1] < 1e-09 else s[0] / s[-1]  # noqa: PLR2004
        is_unstable = cond_a > self.thresholds.max_condition_number_threshold
        if is_unstable:
            return False, np.full(3, np.nan)

        world_in_camera_se3 = camera_in_world_se3.inverse()
        feat_in_camera_vec = world_in_camera_se3 @ feat_in_world_vec

        depth = feat_in_camera_vec[2]

        too_close = depth < self.thresholds.depth_min_threshold
        too_far = depth > self.thresholds.depth_max_threshold
        is_nan = np.isnan(depth)
        is_bad = too_close or too_far or is_nan
        if is_bad:
            return False, np.full(3, np.nan)

        return True, feat_in_world_vec

    def compute_feature_linear_system_update(
        self, feature: Feature, timestamp: float, body_in_world_se3: SE3
    ) -> tuple[Matrix, Vector]:
        """Compute the linear system update for a feature."""
        uv_list = feature.get_uv_by_timestamp(timestamp)

        delta_a = np.zeros((3, 3))
        delta_b = np.zeros(3)

        for cam_id, u, v in uv_list:
            k_matrix_inv = self.k_left_inv if cam_id == 0 else self.k_right_inv
            camera_in_body_se3 = self.cam0_in_body if cam_id == 0 else self.cam1_in_body

            camera_in_world_se3 = body_in_world_se3 * camera_in_body_se3
            camera_in_world_rot = camera_in_world_se3.rotation().as_matrix()
            camera_in_world_vec = camera_in_world_se3.translation()

            pixel_homog = np.array([u, v, 1])
            uv_norm = k_matrix_inv @ pixel_homog
            b_i = np.array([uv_norm[0], uv_norm[1], 1])
            b_i = camera_in_world_rot @ b_i
            b_i = b_i / np.linalg.norm(b_i)
            b_perp = skew(b_i)
            a_i = b_perp.T @ b_perp
            delta_a += a_i
            delta_b += a_i @ camera_in_world_vec

        return delta_a, delta_b

    @classmethod
    def from_euroc_config(cls, euroc_config: EurocConfig) -> "FeatureTriangulation":
        """Create a feature triangulation module from a Euroc configuration."""
        return cls(
            euroc_config.k_matricies(),
            euroc_config.stereo.baseline,
            euroc_config.body_sensor_transforms(),
        )

    @classmethod
    def from_stereo_camera_dto(cls, stereo_camera_dto: StereoCameraDto) -> "FeatureTriangulation":
        """Create a feature triangulation module from a StereoCameraDto."""
        k_matricies = (stereo_camera_dto.stereo_k, stereo_camera_dto.cam0_k, stereo_camera_dto.cam1_k)
        body_sensor_transforms = (stereo_camera_dto.T_body_cam0, stereo_camera_dto.T_body_cam1)
        baseline = stereo_camera_dto.baseline
        return cls(
            k_matricies,
            baseline,
            body_sensor_transforms,
        )
