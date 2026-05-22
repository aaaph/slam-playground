import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.loop_closure.vpr_3d3d_estimator import K3D3DEstimator, K3D3DResult
from core.loop_closure.vpr_frame import VPRFrame, VPRGeometrySchema
from core.transformations.special_euclidian_3_dim import SE3


def make_vpr_frame(frame_id: int, points_xyz: np.ndarray) -> VPRFrame:
    """Make a VPR frame with 3D geometry for K3D3D tests."""
    geometry = np.full((points_xyz.shape[0], VPRGeometrySchema.count()), np.nan, dtype=np.float32)
    geometry[:, VPRGeometrySchema.POINT_X : VPRGeometrySchema.POINT_Z + 1] = points_xyz.astype(np.float32)
    return VPRFrame(
        frame_id=frame_id,
        kf_id=frame_id + 100,
        timestamp=float(frame_id),
        geometry=geometry,
        descriptors=np.zeros((points_xyz.shape[0], 32), dtype=np.uint8),
    )


def make_matches(count: int) -> list[cv2.DMatch]:
    """Make deterministic descriptor matches."""
    return [cv2.DMatch(_queryIdx=index, _trainIdx=index, _distance=0.0) for index in range(count)]


@pytest.fixture
def estimator() -> K3D3DEstimator:
    """Create a deterministic K3D3D estimator."""
    k3d3d_estimator = K3D3DEstimator.default_factory()
    k3d3d_estimator.rigid3d_ransac_iterations = 200
    k3d3d_estimator.rigid3d_ransac_threshold_m = 1e-4
    return k3d3d_estimator


class TestK3D3DEstimator:
    """Unit tests for K3D3DEstimator."""

    def test_estimate_query_pose_should_estimate_reference_t_query(self, estimator: K3D3DEstimator) -> None:
        """Estimator should recover ref_point = reference_T_query * query_point."""
        query_points = np.array(
            [
                [0.0, 0.0, 3.0],
                [1.0, 0.0, 3.2],
                [0.0, 1.0, 3.4],
                [0.5, 0.2, 4.0],
                [-0.4, 0.8, 4.5],
                [1.1, -0.2, 5.0],
            ],
            dtype=np.float64,
        )
        reference_t_query = SE3(
            Rotation.from_euler("xyz", [0.02, -0.03, 0.04]),
            np.array([0.2, -0.1, 0.3], dtype=np.float64),
        )
        reference_points = (
            query_points @ reference_t_query.rotation().as_matrix().T + reference_t_query.translation()
        )
        query_frame = make_vpr_frame(1, query_points)
        reference_frame = make_vpr_frame(2, reference_points)

        result = estimator.estimate_query_pose(query_frame, reference_frame, make_matches(query_points.shape[0]))

        assert result.success
        assert result.matches_count == query_points.shape[0]
        assert result.inliers_count == query_points.shape[0]
        np.testing.assert_allclose(result.reference_t_query.as_matrix(), reference_t_query.as_matrix(), atol=1e-6)
        assert result.median_inlier_residual == pytest.approx(0.0, abs=1e-6)

    def test_estimate_query_pose_should_reject_too_few_matches(self, estimator: K3D3DEstimator) -> None:
        """Estimator should reject inputs below the minimal 3D-3D sample size."""
        query_points = np.array([[0.0, 0.0, 3.0], [1.0, 0.0, 3.0]], dtype=np.float64)
        reference_points = query_points + np.array([0.1, 0.0, 0.0], dtype=np.float64)

        result = estimator.estimate_query_pose(
            make_vpr_frame(1, query_points),
            make_vpr_frame(2, reference_points),
            make_matches(query_points.shape[0]),
        )

        assert not result.success
        assert result.reason == "Not enough matches"
        assert result.matches_count == 0
        assert result.inliers_count == 0
        assert result.residuals.shape == (0,)

    def test_estimate_query_pose_should_reject_too_few_valid_points(self, estimator: K3D3DEstimator) -> None:
        """Estimator should reject when too few matched rows have valid stereo points."""
        query_points = np.array(
            [
                [0.0, 0.0, 3.0],
                [1.0, 0.0, 3.0],
                [0.0, 1.0, -1.0],
                [0.5, 0.2, np.nan],
            ],
            dtype=np.float64,
        )
        reference_points = np.array(
            [
                [0.1, 0.0, 3.0],
                [1.1, 0.0, 3.0],
                [0.1, 1.0, 3.0],
                [0.6, 0.2, 3.0],
            ],
            dtype=np.float64,
        )

        result = estimator.estimate_query_pose(
            make_vpr_frame(1, query_points),
            make_vpr_frame(2, reference_points),
            make_matches(query_points.shape[0]),
        )

        assert not result.success
        assert result.reason == "Not enough valid matches"
        assert result.matches_count == 2
        assert result.inlier_mask.shape == (2,)
        assert result.residuals.shape == (2,)

    def test_estimate_query_pose_should_return_valid_matches_aligned_with_inlier_mask(
        self, estimator: K3D3DEstimator
    ) -> None:
        """Returned matches and inlier mask should both be aligned after invalid 3D rows are removed."""
        query_points = np.array(
            [
                [0.0, 0.0, 3.0],
                [1.0, 0.0, 3.0],
                [0.0, 1.0, 3.0],
                [0.0, 0.0, -1.0],
                [0.5, 0.2, 4.0],
            ],
            dtype=np.float64,
        )
        reference_points = query_points + np.array([0.1, 0.0, 0.0], dtype=np.float64)
        reference_points[3] = np.array([0.1, 0.0, 3.0], dtype=np.float64)

        result = estimator.estimate_query_pose(
            make_vpr_frame(1, query_points),
            make_vpr_frame(2, reference_points),
            make_matches(query_points.shape[0]),
        )

        assert result.success
        assert result.matches_count == 4
        assert result.inlier_mask.shape == (4,)
        assert result.residuals.shape == (4,)
        assert [match.queryIdx for match in result.matches] == [0, 1, 2, 4]

    def test_estimate_query_pose_should_reject_3d_outlier_with_ransac(self, estimator: K3D3DEstimator) -> None:
        """RANSAC should keep the dominant rigid transform and mark a 3D outlier."""
        estimator.rigid3d_ransac_threshold_m = 0.05
        query_points = np.array(
            [
                [0.0, 0.0, 3.0],
                [1.0, 0.0, 3.0],
                [0.0, 1.0, 3.0],
                [0.5, 0.2, 4.0],
                [-0.2, 0.6, 4.5],
                [1.2, -0.1, 5.0],
            ],
            dtype=np.float64,
        )
        translation = np.array([0.2, -0.1, 0.3], dtype=np.float64)
        reference_points = query_points + translation
        reference_points[-1] += np.array([2.0, 0.0, 0.0], dtype=np.float64)

        result = estimator.estimate_query_pose(
            make_vpr_frame(1, query_points),
            make_vpr_frame(2, reference_points),
            make_matches(query_points.shape[0]),
        )

        assert result.success
        assert result.matches_count == query_points.shape[0]
        assert result.inliers_count == query_points.shape[0] - 1
        assert not result.inlier_mask[-1]
        np.testing.assert_allclose(result.reference_t_query.translation(), translation, atol=1e-6)

    def test_k3d3d_result_accepted_should_apply_all_thresholds(self) -> None:
        """Accepted should combine success, inlier count, inlier ratio, and residual gates."""
        result = K3D3DResult(
            success=True,
            reason="Success",
            reference_t_query=SE3.identity(),
            matches=make_matches(4),
            inlier_mask=np.array([True, True, True, False]),
            residuals=np.array([0.0, 0.01, 0.02, 1.0], dtype=np.float64),
        )

        assert result.accepted(min_inliers_count=3, min_inliers_ratio=0.75, max_median_residual=0.02)
        assert not result.accepted(min_inliers_count=4, min_inliers_ratio=0.75, max_median_residual=0.02)
        assert not result.accepted(min_inliers_count=3, min_inliers_ratio=0.8, max_median_residual=0.02)
        assert not result.accepted(min_inliers_count=3, min_inliers_ratio=0.75, max_median_residual=0.005)
