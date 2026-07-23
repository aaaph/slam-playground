from typing import NamedTuple, Self

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
    pixel_sigma_px: float = 20.0
    disparity_sigma_px: float = 0.75


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

    FEAT_ID = 0
    X = 1
    Y = 2
    Z = 3
    STATUS = 4
    COV_XX = 5
    COV_XY = 6
    COV_XZ = 7
    COV_YX = 8
    COV_YY = 9
    COV_YZ = 10
    COV_ZX = 11
    COV_ZY = 12
    COV_ZZ = 13
    DEPTH_SIGMA = 14
    LEFT_U = 15
    LEFT_V = 16
    RIGHT_U = 17
    RIGHT_V = 18

    COV = slice(COV_XX, COV_ZZ + 1)
    XYZ = slice(X, Z + 1)
    LEFT_UV = slice(LEFT_U, LEFT_V + 1)
    RIGHT_UV = slice(RIGHT_U, RIGHT_V + 1)

    @classmethod
    def count(cls) -> int:
        """Return the number of columns in the schema."""
        return cls.RIGHT_V + 1


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
            points_cam0 = np.column_stack((x, y, z))
            covariance = self._stereo_covariance_batch(points_cam0, disp)

        bad_parallax = disp < self.thresholds.disparity_min_threshold
        too_close = z < self.thresholds.depth_min_threshold
        too_far = z > self.thresholds.depth_max_threshold
        bad_vertical_shift = vertical_shift > self.thresholds.vertical_shift_threshold
        non_finite_xyz = ~np.isfinite(x) | ~np.isfinite(y) | ~np.isfinite(z)
        non_finite_covariance = ~np.all(np.isfinite(covariance), axis=1)

        bad_feat_mask = (
            ~finite_uv
            | bad_parallax
            | too_close
            | too_far
            | bad_vertical_shift
            | non_finite_xyz
            | non_finite_covariance
        )

        status = np.logical_not(bad_feat_mask).astype(np.int32)

        batch_triangulation = np.full(
            (stereo_tensor.shape[0], StereoTriangulationSchema.count()), np.nan, dtype=np.float32
        )
        batch_triangulation[:, StereoTriangulationSchema.FEAT_ID] = ids
        batch_triangulation[:, StereoTriangulationSchema.STATUS] = status
        batch_triangulation[:, StereoTriangulationSchema.LEFT_UV] = left_uv
        batch_triangulation[:, StereoTriangulationSchema.RIGHT_UV] = right_uv
        good_feat_mask = ~bad_feat_mask
        batch_triangulation[good_feat_mask, StereoTriangulationSchema.X] = x[good_feat_mask]
        batch_triangulation[good_feat_mask, StereoTriangulationSchema.Y] = y[good_feat_mask]
        batch_triangulation[good_feat_mask, StereoTriangulationSchema.Z] = z[good_feat_mask]
        batch_triangulation[good_feat_mask, StereoTriangulationSchema.COV] = covariance[good_feat_mask]
        batch_triangulation[good_feat_mask, StereoTriangulationSchema.DEPTH_SIGMA] = np.sqrt(
            np.maximum(batch_triangulation[good_feat_mask, StereoTriangulationSchema.COV_ZZ], 0.0)
        )
        return good_feat_mask, batch_triangulation

    def _stereo_covariance_batch(
        self,
        points_cam0: NDArray[np.float64],
        disparity: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute cam0-frame covariance for rectified stereo triangulation."""
        x = points_cam0[:, 0]
        y = points_cam0[:, 1]
        z = points_cam0[:, 2]
        fx = self.k_stereo[0, 0]
        fy = self.k_stereo[1, 1]
        baseline = self.baseline
        pixel_variance = self.thresholds.pixel_sigma_px**2
        disparity_variance = self.thresholds.disparity_sigma_px**2

        # Jacobian: J = ∂p / ∂m -> local sensitivity of p to m changes
        # p = [x, y, z]
        # m = [ul, vl, disp]
        # d = ul - ur
        jac_x_ul = baseline / disparity
        jac_y_vl = fx * baseline / (fy * disparity)
        jac_x_d = -x / disparity
        jac_y_d = -y / disparity
        jac_z_d = -z / disparity

        cov_xx = jac_x_ul**2 * pixel_variance + jac_x_d**2 * disparity_variance
        cov_yy = jac_y_vl**2 * pixel_variance + jac_y_d**2 * disparity_variance
        cov_zz = jac_z_d**2 * disparity_variance
        cov_xy = jac_x_d * jac_y_d * disparity_variance
        cov_xz = jac_x_d * jac_z_d * disparity_variance
        cov_yz = jac_y_d * jac_z_d * disparity_variance

        covariance = np.full((disparity.shape[0], 9), np.nan, dtype=np.float64)
        covariance[:, StereoTriangulationSchema.COV_XX - StereoTriangulationSchema.COV_XX] = cov_xx
        covariance[:, StereoTriangulationSchema.COV_XY - StereoTriangulationSchema.COV_XX] = cov_xy
        covariance[:, StereoTriangulationSchema.COV_XZ - StereoTriangulationSchema.COV_XX] = cov_xz
        covariance[:, StereoTriangulationSchema.COV_YX - StereoTriangulationSchema.COV_XX] = cov_xy
        covariance[:, StereoTriangulationSchema.COV_YY - StereoTriangulationSchema.COV_XX] = cov_yy
        covariance[:, StereoTriangulationSchema.COV_YZ - StereoTriangulationSchema.COV_XX] = cov_yz
        covariance[:, StereoTriangulationSchema.COV_ZX - StereoTriangulationSchema.COV_XX] = cov_xz
        covariance[:, StereoTriangulationSchema.COV_ZY - StereoTriangulationSchema.COV_XX] = cov_yz
        covariance[:, StereoTriangulationSchema.COV_ZZ - StereoTriangulationSchema.COV_XX] = cov_zz
        return covariance

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
