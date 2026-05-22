import cv2
import numpy as np
import pytest

from core.camera_model.stereo_camera_ctx import StereoContext
from core.loop_closure.vpr_frame import VPRFrame, VPRGeometrySchema
from core.loop_closure.vpr_pnp_estimator import VPRPNPEstimator
from core.transformations.special_euclidian_3_dim import SE3


def make_vpr_frame(frame_id: int, left_uv: np.ndarray, right_uv: np.ndarray, points_xyz: np.ndarray) -> VPRFrame:
    """Make a VPR frame with geometry for PnP estimator tests."""
    geometry = np.full((left_uv.shape[0], VPRGeometrySchema.count()), np.nan, dtype=np.float32)
    geometry[:, VPRGeometrySchema.LEFT_U : VPRGeometrySchema.LEFT_V + 1] = left_uv
    geometry[:, VPRGeometrySchema.RIGHT_U : VPRGeometrySchema.RIGHT_V + 1] = right_uv
    geometry[:, VPRGeometrySchema.POINT_X : VPRGeometrySchema.POINT_Z + 1] = points_xyz
    return VPRFrame(
        frame_id=frame_id,
        kf_id=frame_id + 100,
        timestamp=float(frame_id),
        geometry=geometry,
        descriptors=np.zeros((left_uv.shape[0], 32), dtype=np.uint8),
    )


def project(reference_t_query: SE3, reference_points: np.ndarray, k_matrix: np.ndarray) -> np.ndarray:
    """Project reference-frame points into query image coordinates."""
    query_t_reference = reference_t_query.inverse()
    query_points = np.array([query_t_reference.act_on_vector(point) for point in reference_points])
    return np.column_stack(
        [
            k_matrix[0, 0] * query_points[:, 0] / query_points[:, 2] + k_matrix[0, 2],
            k_matrix[1, 1] * query_points[:, 1] / query_points[:, 2] + k_matrix[1, 2],
        ]
    )


class TestVPRPNPEstimator:
    """Unit tests for VPRPNPEstimator."""

    @pytest.fixture
    def stereo_ctx(self) -> StereoContext:
        """Create a stereo camera DTO."""
        return StereoContext(
            resolution=(100, 100),
            stereo_k=np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]),
            cam0_k=np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]),
            cam1_k=np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]),
            baseline=1.0,
            cam0_in_body_se3=SE3.identity(),
            cam1_in_body_se3=SE3.identity(),
        )

    @pytest.fixture
    def vpr_pnp_estimator(self, stereo_ctx: StereoContext) -> VPRPNPEstimator:
        """Create a VPR PnP estimator."""
        return VPRPNPEstimator(stereo_ctx)

    def test_reprojection_errors_should_project_reference_points_into_query_image(
        self, vpr_pnp_estimator: VPRPNPEstimator
    ) -> None:
        """Reprojection errors should be zero for exact reference_T_query and observations."""
        reference_t_query = SE3.from_quat_and_translation(
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            translation=np.array([0.1, -0.2, 0.3]),
        )
        reference_points = np.array(
            [
                [0.1, -0.1, 3.0],
                [0.3, 0.2, 4.0],
                [-0.4, 0.3, 5.0],
            ],
            dtype=np.float64,
        )
        query_t_reference = reference_t_query.inverse()
        query_points = np.array([query_t_reference.act_on_vector(point) for point in reference_points])
        query_uv = np.column_stack(
            [
                1000.0 * query_points[:, 0] / query_points[:, 2],
                1000.0 * query_points[:, 1] / query_points[:, 2],
            ]
        )

        errors = vpr_pnp_estimator.reprojection_errors(reference_t_query, reference_points, query_uv)

        np.testing.assert_allclose(errors, np.zeros(reference_points.shape[0]), atol=1e-9)

    def test_reprojection_errors_should_mark_points_behind_query_camera_as_invalid(
        self, vpr_pnp_estimator: VPRPNPEstimator
    ) -> None:
        """Points behind the query camera should get infinite reprojection error."""
        reference_points = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
        query_uv = np.array([[0.0, 0.0]], dtype=np.float64)

        errors = vpr_pnp_estimator.reprojection_errors(SE3.identity(), reference_points, query_uv)

        assert np.isinf(errors[0])

    def test_estimate_reference_t_query_pnp_ransac_should_reject_too_few_points(
        self, vpr_pnp_estimator: VPRPNPEstimator
    ) -> None:
        """PnP RANSAC should reject inputs below the configured minimum point count."""
        points_count = vpr_pnp_estimator.min_pnp_points - 1
        query_uv = np.zeros((points_count, 2), dtype=np.float32)
        reference_points = np.zeros((points_count, 3), dtype=np.float32)
        query_frame = make_vpr_frame(
            1,
            query_uv,
            np.full_like(query_uv, np.nan),
            np.full_like(reference_points, np.nan),
        )
        reference_frame = make_vpr_frame(
            2,
            np.zeros_like(query_uv),
            np.full_like(query_uv, np.nan),
            reference_points,
        )
        matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0) for i in range(points_count)]

        result = vpr_pnp_estimator.estimate_query_pose(query_frame, reference_frame, matches)

        assert not result.success
        assert result.inliner_mask.shape == (0,)
        assert result.reprojection_errors.shape == (0,)

    def test_estimate_reference_t_query_pnp_ransac_should_handle_opencv_failure(
        self, vpr_pnp_estimator: VPRPNPEstimator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenCV failure should return no pose and a false inlier mask."""
        points_count = vpr_pnp_estimator.min_pnp_points
        query_uv = np.zeros((points_count, 2), dtype=np.float32)
        reference_points = np.column_stack(
            [
                np.linspace(-0.2, 0.2, points_count),
                np.linspace(0.1, 0.3, points_count),
                np.full(points_count, 4.0),
            ]
        ).astype(np.float32)
        query_frame = make_vpr_frame(
            1,
            query_uv,
            np.full_like(query_uv, np.nan),
            np.full_like(reference_points, np.nan),
        )
        reference_frame = make_vpr_frame(
            2,
            np.zeros_like(query_uv),
            np.full_like(query_uv, np.nan),
            reference_points,
        )
        matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0) for i in range(points_count)]

        def fail_solve_pnp_ransac(**_kwargs: object) -> tuple[bool, None, None, None]:
            return False, None, None, None

        monkeypatch.setattr(
            "core.loop_closure.vpr_pnp_estimator.cv2.solvePnPRansac",
            fail_solve_pnp_ransac,
        )

        result = vpr_pnp_estimator.estimate_query_pose(query_frame, reference_frame, matches)

        assert not result.success
        assert result.inliner_mask.shape == (points_count,)
        assert not np.any(result.inliner_mask)
        assert result.reprojection_errors.shape == (0,)

    def test_estimate_query_pose_should_use_match_indices_and_return_reprojection_errors(
        self, vpr_pnp_estimator: VPRPNPEstimator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Estimate should gather query/reference rows through DMatch indices and score the final pose."""
        reference_t_query = SE3.from_quat_and_translation(
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            translation=np.array([0.1, 0.0, 0.0]),
        )
        query_left_uv = np.array(
            [
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
            ],
            dtype=np.float32,
        )
        query_right_uv = query_left_uv - np.array([1.0, 0.0], dtype=np.float32)
        reference_points = np.array(
            [
                [0.1, 0.0, 3.0],
                [0.2, 0.1, 4.0],
                [0.3, 0.2, 5.0],
            ],
            dtype=np.float32,
        )
        query_frame = make_vpr_frame(
            1,
            query_left_uv,
            query_right_uv,
            np.full_like(reference_points, np.nan, dtype=np.float32),
        )
        reference_frame = make_vpr_frame(
            2,
            np.zeros_like(query_left_uv),
            np.full_like(query_left_uv, np.nan, dtype=np.float32),
            reference_points,
        )
        matches = [
            cv2.DMatch(_queryIdx=2, _trainIdx=1, _distance=0.0),
            cv2.DMatch(_queryIdx=0, _trainIdx=2, _distance=0.0),
            cv2.DMatch(_queryIdx=1, _trainIdx=0, _distance=0.0),
        ]
        pnp_mask = np.array([True, False, True])
        pnp_calls: list[tuple[np.ndarray, np.ndarray]] = []
        ba_calls: list[tuple[SE3, np.ndarray]] = []

        def fake_pnp(points: np.ndarray, uv: np.ndarray) -> tuple[SE3, np.ndarray, str]:
            pnp_calls.append((points.copy(), uv.copy()))
            return reference_t_query, pnp_mask, "fake pnp success"

        def fake_ba(pose: SE3, visual_features: np.ndarray) -> SE3:
            ba_calls.append((pose, visual_features.copy()))
            return reference_t_query

        monkeypatch.setattr(vpr_pnp_estimator, "_estimate_reference_t_query_pnp_ransac", fake_pnp)
        monkeypatch.setattr(vpr_pnp_estimator, "_motion_only_ba", fake_ba)

        result = vpr_pnp_estimator.estimate_query_pose(query_frame, reference_frame, matches)

        expected_points = reference_points[[1, 2, 0]].astype(np.float64)
        expected_uv = query_left_uv[[2, 0, 1]].astype(np.float64)
        np.testing.assert_allclose(pnp_calls[0][0], expected_points)
        np.testing.assert_allclose(pnp_calls[0][1], expected_uv)
        np.testing.assert_allclose(ba_calls[0][1][:, 5:8], expected_points[pnp_mask])
        assert result.success
        np.testing.assert_array_equal(result.inliner_mask, pnp_mask)
        np.testing.assert_allclose(
            result.reprojection_errors,
            vpr_pnp_estimator.reprojection_errors(reference_t_query, expected_points, expected_uv),
        )

    def test_motion_only_ba_should_keep_exact_mono_pose(
        self, vpr_pnp_estimator: VPRPNPEstimator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Motion-only BA should optimize a pose with fixed reference landmarks."""
        reference_t_query = SE3.identity()
        reference_points = np.array(
            [
                [0.1, 0.0, 3.0],
                [0.2, 0.1, 4.0],
                [-0.2, 0.1, 5.0],
                [0.0, -0.2, 6.0],
            ],
            dtype=np.float64,
        )
        query_uv = project(reference_t_query, reference_points, vpr_pnp_estimator.stereo_ctx.stereo_k)
        query_right_uv = np.full_like(query_uv, np.nan, dtype=np.float64)
        query_frame = make_vpr_frame(
            1,
            query_uv.astype(np.float32),
            query_right_uv.astype(np.float32),
            np.full_like(reference_points, np.nan, dtype=np.float32),
        )
        reference_frame = make_vpr_frame(
            2,
            np.zeros_like(query_uv, dtype=np.float32),
            np.full_like(query_uv, np.nan, dtype=np.float32),
            reference_points.astype(np.float32),
        )
        matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0) for i in range(reference_points.shape[0])]
        inlier_mask = np.ones(reference_points.shape[0], dtype=bool)

        def exact_pnp(_points: np.ndarray, _uv: np.ndarray) -> tuple[SE3, np.ndarray, str]:
            return reference_t_query, inlier_mask, "fake pnp success"

        monkeypatch.setattr(vpr_pnp_estimator, "_estimate_reference_t_query_pnp_ransac", exact_pnp)

        result = vpr_pnp_estimator.estimate_query_pose(query_frame, reference_frame, matches)

        assert result.success
        np.testing.assert_allclose(result.reference_t_query.as_matrix(), reference_t_query.as_matrix(), atol=1e-7)
        np.testing.assert_allclose(result.reprojection_errors, np.zeros(reference_points.shape[0]), atol=1e-6)
