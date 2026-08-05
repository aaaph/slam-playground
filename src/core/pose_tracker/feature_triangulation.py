from enum import IntEnum
from typing import NamedTuple, Self

import numpy as np
from numpy.typing import NDArray

from core.camera_model.stereo_camera_ctx import StereoContext
from core.feature_tracker.feature_schema import FeatureSchema
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
CameraMatrices = tuple[StereoCameraMatrix, LeftCameraMatrix, RightCameraMatrix]
Baseline = float
BodySensorTransform = SE3  # shape: (4, 4)
LeftCameraInBodyTransform = BodySensorTransform
RightCameraInBodyTransform = BodySensorTransform
BodySensorTransforms = tuple[LeftCameraInBodyTransform, RightCameraInBodyTransform]
Matrix = NDArray[np.float64]  # shape: (3, 3)
Vector = NDArray[np.float64]  # shape: (3,)


class StereoTriangulationSchema:
    """Schema for stereo triangulation."""

    FEAT_ID = FeatureSchema.FEAT_ID
    TIMESTAMP = FeatureSchema.TIMESTAMP
    LEFT_U = FeatureSchema.LEFT_U
    LEFT_V = FeatureSchema.LEFT_V
    RIGHT_U = FeatureSchema.RIGHT_U
    RIGHT_V = FeatureSchema.RIGHT_V
    LIFECYCLE = FeatureSchema.LIFECYCLE
    AGE = FeatureSchema.AGE
    STEREO_SCORE = FeatureSchema.STEREO_SCORE
    FRAME_PIXEL_DISPLACEMENT = FeatureSchema.FRAME_PIXEL_DISPLACEMENT
    LEFT_BEARING_X = FeatureSchema.LEFT_BEARING_X
    LEFT_BEARING_Y = FeatureSchema.LEFT_BEARING_Y
    LEFT_BEARING_Z = FeatureSchema.LEFT_BEARING_Z

    STEREO_X = FeatureSchema.count()
    STEREO_Y = STEREO_X + 1
    STEREO_Z = STEREO_Y + 1
    STEREO_STATUS = STEREO_Z + 1

    TRACKER = slice(FEAT_ID, FeatureSchema.count())
    XYZ = slice(STEREO_X, STEREO_Z + 1)
    LEFT_UV = slice(LEFT_U, LEFT_V + 1)
    RIGHT_UV = slice(RIGHT_U, RIGHT_V + 1)
    LEFT_BEARING = slice(LEFT_BEARING_X, LEFT_BEARING_Z + 1)

    @classmethod
    def count(cls) -> int:
        """Return the number of columns in the schema."""
        return cls.STEREO_STATUS + 1


class StereoTriangulationStatus(IntEnum):
    """Stereo triangulation row status."""

    UNTRACKED = 0
    BAD_STEREO = 1
    TRIANGULATED = 2


class FeatureTriangulation:
    """Feature triangulation module."""

    def __init__(
        self,
        k_matrices: CameraMatrices,
        baseline: Baseline,
        body_sensor_transforms: BodySensorTransforms,
        thresholds: FeatureTriangulationThresholds = _DEFAULT_THRESHOLDS,
    ) -> None:
        """Initialize the feature triangulation module."""
        self.logger = spawn_logger(app="feature_triangulation")
        self.k_stereo, self.k_left, self.k_right = k_matrices
        self.baseline = baseline
        self.thresholds = thresholds
        self.cam0_in_body, self.cam1_in_body = body_sensor_transforms
        self.k_stereo_inv = np.linalg.inv(self.k_stereo)
        self.k_left_inv = np.linalg.inv(self.k_left)
        self.k_right_inv = np.linalg.inv(self.k_right)
        self.body_in_cam0 = self.cam0_in_body.inverse()
        self.body_in_cam1 = self.cam1_in_body.inverse()

    def make_initial_guess_by_stereo_batch(
        self, stereo_tensor: NDArray[np.float32]
    ) -> tuple[NDArray[np.bool_], NDArray[np.float32]]:
        """Make an initial guess for a feature over last stereo pair."""
        # stereo_tensor: (N, 5) - feat_id, left_u, left_v, right_u, right_v
        ids = stereo_tensor[:, 0].astype(np.int32)
        left_uv = stereo_tensor[:, 1:3].astype(np.float64)
        right_uv = stereo_tensor[:, 3:5].astype(np.float64)

        fx = self.k_stereo[0, 0]
        fy = self.k_stereo[1, 1]
        cx = self.k_stereo[0, 2]
        cy = self.k_stereo[1, 2]
        baseline = self.baseline

        disp = left_uv[:, 0] - right_uv[:, 0]
        finite_uv = np.all(np.isfinite(stereo_tensor[:, 1:5]), axis=1)
        vertical_shift = np.abs(left_uv[:, 1] - right_uv[:, 1])

        with np.errstate(divide="ignore", invalid="ignore"):
            z = fx * baseline / disp
            x = (left_uv[:, 0] - cx) * z / fx
            y = (left_uv[:, 1] - cy) * z / fy

        bad_parallax = disp < self.thresholds.disparity_min_threshold
        too_close = z < self.thresholds.depth_min_threshold
        too_far = z > self.thresholds.depth_max_threshold
        bad_vertical_shift = vertical_shift > self.thresholds.vertical_shift_threshold
        non_finite_xyz = ~np.isfinite(x) | ~np.isfinite(y) | ~np.isfinite(z)

        bad_feat_mask = ~finite_uv | bad_parallax | too_close | too_far | bad_vertical_shift | non_finite_xyz

        status = np.where(
            bad_feat_mask,
            StereoTriangulationStatus.BAD_STEREO.value,
            StereoTriangulationStatus.TRIANGULATED.value,
        ).astype(np.int32)

        batch_triangulation = np.full(
            (stereo_tensor.shape[0], StereoTriangulationSchema.count()), np.nan, dtype=np.float32
        )
        batch_triangulation[:, StereoTriangulationSchema.FEAT_ID] = ids
        batch_triangulation[:, StereoTriangulationSchema.STEREO_STATUS] = status
        batch_triangulation[:, StereoTriangulationSchema.LEFT_UV] = left_uv
        batch_triangulation[:, StereoTriangulationSchema.RIGHT_UV] = right_uv
        good_feat_mask = ~bad_feat_mask
        batch_triangulation[good_feat_mask, StereoTriangulationSchema.STEREO_X] = x[good_feat_mask]
        batch_triangulation[good_feat_mask, StereoTriangulationSchema.STEREO_Y] = y[good_feat_mask]
        batch_triangulation[good_feat_mask, StereoTriangulationSchema.STEREO_Z] = z[good_feat_mask]
        return good_feat_mask, batch_triangulation

    @classmethod
    def from_stereo_camera_ctx(cls, stereo_ctx: StereoContext) -> Self:
        """Create a feature triangulation module from a StereoCameraCtx."""
        k_matrices = (stereo_ctx.stereo_k, stereo_ctx.cam0_k, stereo_ctx.cam1_k)
        body_sensor_transforms = (stereo_ctx.cam0_in_body_se3, stereo_ctx.cam1_in_body_se3)
        baseline = stereo_ctx.baseline
        return cls(
            k_matrices,
            baseline,
            body_sensor_transforms,
        )
