from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from core.loop_closure.vpr_frame import VPRFrame
from core.transformations.special_euclidian_3_dim import SE3

type PointsXYZ = NDArray[np.float64]


@dataclass(frozen=True, slots=True, repr=False)
class K3D3DResult:
    """VPR 3D-3D result."""

    success: bool
    reason: str
    reference_t_query: SE3
    matches: list[cv2.DMatch]
    inlier_mask: np.ndarray
    residuals: np.ndarray

    @property
    def matches_count(self) -> int:
        """Get valid 3D descriptor matches count."""
        return len(self.matches)

    @property
    def inliers_count(self) -> int:
        """Get rigid-transform inliers count."""
        return int(np.count_nonzero(self.inlier_mask))

    @property
    def inliers_ratio(self) -> float:
        """Get rigid-transform inliers ratio."""
        if self.matches_count == 0:
            return 0.0
        return self.inliers_count / self.matches_count

    @property
    def median_inlier_residual(self) -> float:
        """Get median residual over rigid-transform inliers."""
        if self.inliers_count == 0:
            return np.inf
        return float(np.median(self.residuals[self.inlier_mask]))

    def __repr__(self) -> str:
        """Return a string representation of the K3D3DResult."""
        inliers_str = f"{self.inliers_count}/{self.matches_count}"
        return (
            f"K3D3DResult(success={self.success}, reason={self.reason}, "
            f"reference_t_query={self.reference_t_query}, median_residual={self.median_inlier_residual}, "
            f"inliers={inliers_str})"
        )

    def accepted(self, min_inliers_count: int, min_inliers_ratio: float, max_median_residual: float) -> bool:
        """Check if the K3D3DResult is accepted."""
        return (
            self.success
            and self.inliers_count >= min_inliers_count
            and self.inliers_ratio >= min_inliers_ratio
            and self.median_inlier_residual <= max_median_residual
        )

    @classmethod
    def failed(cls, reason: str) -> "K3D3DResult":
        """Create a failed VPR 3D-3D result."""
        return cls(
            success=False,
            reason=reason,
            reference_t_query=SE3.identity(),
            matches=[],
            inlier_mask=np.empty((0,), dtype=bool),
            residuals=np.empty((0,), dtype=np.float64),
        )

    @classmethod
    def failed_with_matches(cls, reason: str, matches: list[cv2.DMatch]) -> "K3D3DResult":
        """Create a failed VPR 3D-3D result with matches."""
        return cls(
            success=False,
            reason=reason,
            reference_t_query=SE3.identity(),
            matches=matches,
            inlier_mask=np.zeros((len(matches),), dtype=bool),
            residuals=np.full((len(matches),), np.inf, dtype=np.float64),
        )

    @classmethod
    def empty(cls) -> "K3D3DResult":
        """Create an empty K3D3DResult."""
        return cls(
            success=False,
            reason="Empty",
            reference_t_query=SE3.identity(),
            matches=[],
            inlier_mask=np.empty((0,), dtype=bool),
            residuals=np.empty((0,), dtype=np.float64),
        )


class K3D3DEstimator:
    """VPR 3D-3D RANSAC estimator."""

    def __init__(self) -> None:
        """Initialize the VPR 3D-3D estimator."""
        self.min_3d3d_points = 3
        self.rigid3d_ransac_seed = 42
        self.rigid3d_ransac_iterations = 100
        self.rigid3d_ransac_threshold_m = 0.15

    @classmethod
    def default_factory(cls) -> "K3D3DEstimator":
        """Create a default K3D3DEstimator."""
        return cls()

    @staticmethod
    def _valid_3d_points(points: PointsXYZ) -> np.ndarray:
        """Check whether 3D points are finite and in front of the camera."""
        return np.all(np.isfinite(points), axis=1) & (points[:, 2] > 0.0)

    @staticmethod
    def _estimate_rigid_transform(query_points: PointsXYZ, reference_points: PointsXYZ) -> SE3:
        """Estimate the rigid transform ref_point = R * query_point + t by SVD."""
        query_centroid = np.mean(query_points, axis=0)
        reference_centroid = np.mean(reference_points, axis=0)
        query_centered = query_points - query_centroid
        reference_centered = reference_points - reference_centroid

        covariance = query_centered.T @ reference_centered
        u_matrix, _singular_values, vt_matrix = np.linalg.svd(covariance)
        rotation = vt_matrix.T @ u_matrix.T
        if np.linalg.det(rotation) < 0:
            vt_matrix[-1, :] *= -1
            rotation = vt_matrix.T @ u_matrix.T
        translation = reference_centroid - rotation @ query_centroid

        return SE3(Rotation.from_matrix(rotation), translation)

    @staticmethod
    def _rigid_transform_residuals(
        query_points: PointsXYZ,
        reference_points: PointsXYZ,
        reference_t_query: SE3,
    ) -> np.ndarray:
        """Compute Euclidean residuals for ref_point ~= T_ref_query * query_point."""
        transformed_query = (
            query_points @ reference_t_query.rotation().as_matrix().T + reference_t_query.translation()
        )
        return np.linalg.norm(reference_points - transformed_query, axis=1)

    def estimate_query_pose(
        self, query_frame: VPRFrame, reference_frame: VPRFrame, matches: list[cv2.DMatch]
    ) -> K3D3DResult:
        """Estimate the query pose."""
        if len(matches) < self.min_3d3d_points:
            return K3D3DResult.failed("Not enough matches")

        query_points = query_frame.points_xyz[[match.queryIdx for match in matches]].astype(np.float64)
        reference_points = reference_frame.points_xyz[[match.trainIdx for match in matches]].astype(np.float64)

        valid_mask = self._valid_3d_points(query_points) & self._valid_3d_points(reference_points)
        valid_matches = [match for match, is_valid in zip(matches, valid_mask, strict=False) if is_valid]
        query_points = query_points[valid_mask]
        reference_points = reference_points[valid_mask]

        if query_points.shape[0] < self.min_3d3d_points:
            return K3D3DResult.failed_with_matches("Not enough valid matches", valid_matches)

        reference_t_query = self._estimate_rigid_transform_ransac(query_points, reference_points)
        if reference_t_query is None:
            valid_matches = [match for match, is_valid in zip(matches, valid_mask, strict=False) if is_valid]
            return K3D3DResult.failed_with_matches(
                "Failed to estimate reference_T_query with 3D-3D RANSAC", valid_matches
            )

        residuals = self._rigid_transform_residuals(query_points, reference_points, reference_t_query)
        inliers = residuals < self.rigid3d_ransac_threshold_m
        if int(np.count_nonzero(inliers)) >= self.min_3d3d_points:
            reference_t_query = self._estimate_rigid_transform(
                query_points[inliers],
                reference_points[inliers],
            )
            residuals = self._rigid_transform_residuals(query_points, reference_points, reference_t_query)
            inliers = residuals < self.rigid3d_ransac_threshold_m

        return K3D3DResult(
            success=True,
            reason="Success",
            reference_t_query=reference_t_query,
            matches=valid_matches,
            inlier_mask=inliers,
            residuals=residuals,
        )

    def _estimate_rigid_transform_ransac(self, query_points: PointsXYZ, reference_points: PointsXYZ) -> SE3 | None:
        """Estimate a rigid transform with deterministic 3-point RANSAC."""
        if query_points.shape[0] < self.min_3d3d_points:
            return None

        rng = np.random.default_rng(self.rigid3d_ransac_seed)
        best_inliers = np.empty((0,), dtype=bool)
        best_median_residual = np.inf
        best_transform = None

        for _ in range(self.rigid3d_ransac_iterations):
            sample_indices = rng.choice(query_points.shape[0], size=self.min_3d3d_points, replace=False)
            reference_t_query = self._estimate_rigid_transform(
                query_points[sample_indices],
                reference_points[sample_indices],
            )
            residuals = self._rigid_transform_residuals(query_points, reference_points, reference_t_query)
            inliers = residuals < self.rigid3d_ransac_threshold_m
            inliers_count = int(np.count_nonzero(inliers))
            median_residual = np.inf if inliers_count == 0 else float(np.median(residuals[inliers]))

            best_choice = inliers_count > int(np.count_nonzero(best_inliers)) or (
                inliers_count == int(np.count_nonzero(best_inliers)) and median_residual < best_median_residual
            )
            if best_choice:
                best_inliers = inliers
                best_median_residual = median_residual
                best_transform = reference_t_query

        if best_transform is None:
            return None
        if int(np.count_nonzero(best_inliers)) >= self.min_3d3d_points:
            return self._estimate_rigid_transform(query_points[best_inliers], reference_points[best_inliers])
        return best_transform
