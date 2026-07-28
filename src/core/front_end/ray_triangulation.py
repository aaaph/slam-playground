from enum import Enum
from typing import NamedTuple, Self

import numpy as np
from numpy.typing import NDArray

from core.camera_model.stereo_camera_ctx import StereoContext

type FeatureId = NDArray[np.int32]  # shape: (1,)
type UV = NDArray[np.float64]  # shape: (2,)
type CameraIndex = int

type PoseMatrix = NDArray[np.float64]  # shape: (4, 4)


class RayTriangulationThresholds(NamedTuple):
    """Thresholds for ray triangulation."""

    min_depth: float = 0.15
    max_depth: float = 40.0
    max_condition_number: float = 10000.0
    min_singular_value: float = 1e-09
    min_rank: int = 3


class TriangulationStatus(Enum):
    """Status of the triangulation."""

    SUCCESS = 0
    RANK_DEFICIENT = 1
    ILL_CONDITIONED = 2
    INVALID_DEPTH = 3
    LINEAR_SYSTEM_SINGULAR = 4


DEFAULT_THRESHOLDS = RayTriangulationThresholds()


class RayTriangulation:
    """Component for triangulating observations using ray casting."""

    def __init__(
        self, stereo_ctx: StereoContext, thresholds: RayTriangulationThresholds = DEFAULT_THRESHOLDS
    ) -> None:
        """Initialize the ray triangulation component."""
        self._stere_ctx = stereo_ctx
        self._thresholds = thresholds
        self.k_inv = np.linalg.inv(stereo_ctx.stereo_k)
        self.cam0_from_cam1 = (stereo_ctx.cam0_in_body_se3.inverse() * stereo_ctx.cam1_in_body_se3).as_matrix()
        self.baseline = stereo_ctx.baseline
        self.rect0_from_rect1 = np.eye(4, dtype=np.float64)
        self.rect0_from_rect1[0, 3] = self.baseline

    @classmethod
    def default_factory(cls, stereo_ctx: StereoContext) -> Self:
        """Create a default ray triangulation component."""
        return cls(stereo_ctx, DEFAULT_THRESHOLDS)

    def _compute_linear_system(
        self, uv: NDArray[np.float64], anchor_from_camera: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute the linear system for one feature."""
        pixel_homogeneous = np.ones((uv.shape[0], 3), dtype=np.float64)
        pixel_homogeneous[:, :2] = uv
        bearings_camera = pixel_homogeneous @ self.k_inv.T
        bearings_anchor = np.einsum("nij,nj->ni", anchor_from_camera[:, :3, :3], bearings_camera)
        bearings_anchor /= np.linalg.norm(bearings_anchor, axis=1, keepdims=True)

        a_i = np.eye(3, dtype=np.float64)[None, :, :] - np.einsum("ni,nj->nij", bearings_anchor, bearings_anchor)
        a = np.sum(a_i, axis=0)
        b = np.sum(np.einsum("nij,nj->ni", a_i, anchor_from_camera[:, :3, 3]), axis=0)
        return a, b

    def triangulate_feature_observations(
        self, left_uv: UV, right_uv: UV, cam0_poses: PoseMatrix
    ) -> tuple[TriangulationStatus, NDArray[np.float64]]:
        """
        Triangulate observations using ray casting.

        Expecting that left_uv should not contain invalid values. Expecting that right_uv is nullable.

        Args:
            feat_ids: Feature IDs. shape: (N,)
            left_uv: Left UV coordinates. shape: (N, 2)
            right_uv: Right UV coordinates. shape: (N, 2)
            cam0_poses: Camera 0 poses. shape: (N, 4, 4)


        Returns:
            Triangulated points. shape: (N, 3)

        """
        left_num = left_uv.shape[0]

        right_valid_mask = np.all(np.isfinite(right_uv), axis=1)
        right_uv_valid = right_uv[right_valid_mask, :]
        right_num = right_uv_valid.shape[0]

        uv = np.full((left_num + right_num, 2), np.nan, dtype=np.float64)
        poses = np.full((left_num + right_num, 4, 4), np.nan, dtype=np.float64)
        uv[:left_num, :] = left_uv
        poses[:left_num, :, :] = cam0_poses[:left_num, :, :]
        uv[left_num:, :] = right_uv_valid
        poses[left_num:, :, :] = cam0_poses[right_valid_mask, :, :] @ self.rect0_from_rect1

        a, b = self._compute_linear_system(uv, poses)

        try:
            solution, _, rank, s = np.linalg.lstsq(a, b, rcond=None)
            if rank < self._thresholds.min_rank:  # rank is 3
                return TriangulationStatus.RANK_DEFICIENT, np.full((0, 3), np.nan, dtype=np.float64)
            point_in_anchor = solution
        except np.linalg.LinAlgError:
            return TriangulationStatus.LINEAR_SYSTEM_SINGULAR, np.full((0, 3), np.nan, dtype=np.float64)

        cond_a = np.inf if s[-1] < self._thresholds.min_singular_value else s[0] / s[-1]  # < 1e-09 is unstable
        is_unstable = cond_a > self._thresholds.max_condition_number  # > 10000.0 is unstable

        point_in_anchor = np.asarray(point_in_anchor, dtype=np.float64)
        if is_unstable:
            return TriangulationStatus.ILL_CONDITIONED, point_in_anchor

        points_in_camera = np.einsum(
            "nji,nj->ni",
            poses[:, :3, :3],
            point_in_anchor[None, :] - poses[:, :3, 3],
        )
        depths = points_in_camera[:, 2]

        too_close = np.any(depths < self._thresholds.min_depth)  # < 0.15 is invalid
        too_far = np.any(depths > self._thresholds.max_depth)  # > 40.0 is invalid
        non_finite_depth = not np.all(np.isfinite(depths))
        is_invalid_depth = too_close or too_far or non_finite_depth
        # could be issue to check point for all cameras -> could be migrated to check only via anchor camera

        if is_invalid_depth:
            return TriangulationStatus.INVALID_DEPTH, point_in_anchor

        return TriangulationStatus.SUCCESS, point_in_anchor
