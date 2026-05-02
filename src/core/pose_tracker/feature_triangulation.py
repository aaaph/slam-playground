from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from core.camera_model.stereo_camera_ctx import StereoContext
from core.transformations.special_euclidian_3_dim import SE3
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

    def make_initial_guess_by_stereo_batch(self, stereo_tensor: NDArray[np.float32]) -> NDArray[np.float32]:
        """Make an initial guess for a feature over last stereo pair."""
        # stereo_tensor: (N, 5) - feat_id, left_u, left_v, right_u, right_v
        ids = stereo_tensor[:, 0].astype(np.int32)
        left_uv = stereo_tensor[:, 1:3]
        right_uv = stereo_tensor[:, 3:5]

        fx = self.k_stereo[0, 0]
        fy = self.k_stereo[1, 1]
        cx = self.k_stereo[0, 2]
        cy = self.k_stereo[1, 2]
        baseline = self.baseline

        with np.errstate(divide="ignore", invalid="ignore"):
            disp = left_uv[:, 0] - right_uv[:, 0]
        z = fx * baseline / disp
        x = (left_uv[:, 0] - cx) * z / fx
        y = (left_uv[:, 1] - cy) * z / fy

        bad_parallax = disp < self.thresholds.disparity_min_threshold
        too_close = z < self.thresholds.depth_min_threshold
        too_far = z > self.thresholds.depth_max_threshold
        vertical_shift = np.abs(left_uv[:, 1] - right_uv[:, 1]) > self.thresholds.vertical_shift_threshold
        is_nan = np.isnan(disp) | np.isnan(z)

        bad_feat_mask = bad_parallax | too_close | too_far | vertical_shift | is_nan

        status = np.logical_not(bad_feat_mask).astype(np.int32)
        new_tensor = np.column_stack((ids, x, y, z, status))
        new_tensor[bad_feat_mask, 1:4] = np.nan
        return new_tensor

    """def make_linear_triangulation_guess(sel
    f, feature: Feature, camera_in_world_se3: SE3) -> tuple[bool, Vector]:
        Make a linear triangulation guess for a feature.
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
        Compute the linear system update for a feature.
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

        return delta_a, delta_b"""

    @classmethod
    def from_stereo_camera_ctx(cls, stereo_ctx: StereoContext) -> "FeatureTriangulation":
        """Create a feature triangulation module from a StereoCameraCtx."""
        k_matricies = (stereo_ctx.stereo_k, stereo_ctx.cam0_k, stereo_ctx.cam1_k)
        body_sensor_transforms = (stereo_ctx.cam0_in_body_se3, stereo_ctx.cam1_in_body_se3)
        baseline = stereo_ctx.baseline
        return cls(
            k_matricies,
            baseline,
            body_sensor_transforms,
        )
