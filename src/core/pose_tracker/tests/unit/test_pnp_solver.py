import cv2
import numpy as np
import pytest

from core.camera_model.stereo_camera_ctx import StereoContext
from core.pose_tracker.frame_to_frame_pnp_store import PnPMapSchema
from core.pose_tracker.pnp_solver import (
    PnpPoseSolver,
    PnpSolverConfig,
    PnpSolveStatus,
    _MotionOnlyBaResult,
    _PnPRansacResult,
)
from core.transformations.special_euclidian_3_dim import SE3


def make_stereo_ctx() -> StereoContext:
    """Create a stereo camera DTO."""
    k_matrix = np.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]])
    return StereoContext(
        resolution=(640, 480),
        stereo_k=k_matrix,
        cam0_k=k_matrix,
        cam1_k=k_matrix,
        baseline=0.1,
        cam0_in_body_se3=SE3.identity(),
        cam1_in_body_se3=SE3.identity(),
    )


def make_reference_points() -> np.ndarray:
    """Create non-coplanar reference points for PnP tests."""
    return np.array(
        [
            [-0.7, -0.4, 4.0],
            [-0.2, -0.5, 4.3],
            [0.4, -0.3, 4.8],
            [0.8, -0.1, 5.2],
            [-0.6, 0.2, 4.5],
            [-0.1, 0.1, 5.0],
            [0.5, 0.2, 5.4],
            [0.9, 0.4, 5.8],
            [-0.4, 0.6, 6.1],
            [0.1, 0.7, 6.4],
            [0.6, 0.6, 6.8],
            [1.0, 0.8, 7.2],
        ],
        dtype=np.float64,
    )


def project(cam0_in_reference: SE3, reference_points: np.ndarray, k_matrix: np.ndarray) -> np.ndarray:
    """Project reference-frame points into current cam0 image coordinates."""
    reference_in_cam0 = cam0_in_reference.inverse()
    cam0_points = reference_in_cam0.rotation().apply(reference_points) + reference_in_cam0.translation()
    return np.column_stack(
        (
            k_matrix[0, 0] * cam0_points[:, 0] / cam0_points[:, 2] + k_matrix[0, 2],
            k_matrix[1, 1] * cam0_points[:, 1] / cam0_points[:, 2] + k_matrix[1, 2],
        )
    )


def make_visual_features(
    feat_ids: np.ndarray,
    object_points: np.ndarray,
    left_uv: np.ndarray,
    right_uv: np.ndarray | None = None,
) -> np.ndarray:
    """Create the visual feature table consumed by PnpPoseSolver."""
    visual_features = np.full((feat_ids.shape[0], PnPMapSchema.count()), np.nan, dtype=np.float64)
    visual_features[:, PnPMapSchema.FEAT_ID] = feat_ids
    visual_features[:, PnPMapSchema.XYZ] = object_points
    visual_features[:, PnPMapSchema.LEFT_UV] = left_uv
    if right_uv is not None:
        visual_features[:, PnPMapSchema.RIGHT_UV] = right_uv
    return visual_features


class TestPnpPoseSolver:
    """Unit tests for PnpPoseSolver."""

    def test_solve_visual_features_estimates_cam0_in_reference(self) -> None:
        """Solver should recover reference_T_cam0 from exact visual features."""
        stereo_ctx = make_stereo_ctx()
        cam0_in_reference = SE3.from_rpy_xyz(
            rpy=np.array([0.03, -0.02, 0.04]),
            translation=np.array([0.2, -0.1, 0.3]),
        )
        object_points = make_reference_points()
        left_uv = project(cam0_in_reference, object_points, stereo_ctx.stereo_k)
        feat_ids = np.arange(object_points.shape[0], dtype=np.int32)
        visual_features = make_visual_features(feat_ids, object_points, left_uv)
        solver = PnpPoseSolver(stereo_ctx, PnpSolverConfig(motion_only_ba_enabled=False))

        result = solver.solve_visual_features(visual_features)

        assert result.ok
        np.testing.assert_array_equal(result.inlier_feat_ids, feat_ids)
        np.testing.assert_allclose(
            result.cam0_in_reference.translation(),
            cam0_in_reference.translation(),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result.cam0_in_reference.rotation().as_matrix(),
            cam0_in_reference.rotation().as_matrix(),
            atol=1e-6,
        )
        np.testing.assert_allclose(result.reprojection_errors, np.zeros(object_points.shape[0]), atol=2e-5)

    def test_solve_visual_features_passes_rows_to_pnp_without_filtering(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Solver should treat visual features as a strict caller-owned contract."""
        stereo_ctx = make_stereo_ctx()
        object_points = make_reference_points()[:4].astype(np.float64)
        left_uv = project(SE3.identity(), object_points, stereo_ctx.stereo_k)
        feat_ids = np.array([10, 20, 30, 40], dtype=np.int32)
        solver = PnpPoseSolver(
            stereo_ctx,
            PnpSolverConfig(min_points=3, refine_with_lm=False, motion_only_ba_enabled=False),
        )
        pnp_calls = []

        def fake_solve_pnp_ransac(**kwargs: object) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
            object_points_arg = kwargs["objectPoints"]
            image_points_arg = kwargs["imagePoints"]
            assert isinstance(object_points_arg, np.ndarray)
            assert isinstance(image_points_arg, np.ndarray)
            pnp_calls.append((object_points_arg.copy(), image_points_arg.copy()))
            return (
                True,
                np.zeros((3, 1), dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
                np.array([[0], [1], [3]], dtype=np.int32),
            )

        monkeypatch.setattr("core.pose_tracker.pnp_solver.cv2.solvePnPRansac", fake_solve_pnp_ransac)
        visual_features = make_visual_features(feat_ids, object_points, left_uv)

        result = solver.solve_visual_features(visual_features)

        np.testing.assert_allclose(pnp_calls[0][0], object_points)
        np.testing.assert_allclose(pnp_calls[0][1], left_uv)
        assert result.ok
        np.testing.assert_array_equal(result.inlier_feat_ids, np.array([10, 20, 40], dtype=np.int32))
        np.testing.assert_array_equal(result.outlier_feat_ids, np.array([30], dtype=np.int32))

    def test_solve_visual_features_rejects_too_few_rows(self) -> None:
        """Solver should fail before OpenCV when the caller passes too few rows."""
        stereo_ctx = make_stereo_ctx()
        object_points = np.array(
            [
                [0.0, 0.0, 4.0],
                [0.1, 0.1, 4.2],
                [0.2, 0.2, 4.4],
            ],
            dtype=np.float64,
        )
        left_uv = np.array(
            [
                [320.0, 240.0],
                [321.0, 241.0],
                [322.0, 242.0],
            ],
            dtype=np.float64,
        )
        feat_ids = np.arange(object_points.shape[0], dtype=np.int32)
        solver = PnpPoseSolver(stereo_ctx, PnpSolverConfig(motion_only_ba_enabled=False))
        visual_features = make_visual_features(feat_ids, object_points, left_uv)

        result = solver.solve_visual_features(visual_features)

        assert result.status is PnpSolveStatus.NOT_ENOUGH_POINTS
        assert not result.ok
        np.testing.assert_array_equal(result.inlier_mask, np.zeros(feat_ids.shape[0], dtype=bool))
        np.testing.assert_array_equal(result.outlier_feat_ids, feat_ids)

    def test_solve_visual_features_returns_failure_when_opencv_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Solver should convert OpenCV errors into a failed result."""
        stereo_ctx = make_stereo_ctx()
        object_points = make_reference_points()
        left_uv = project(SE3.identity(), object_points, stereo_ctx.stereo_k)
        feat_ids = np.arange(object_points.shape[0], dtype=np.int32)
        solver = PnpPoseSolver(stereo_ctx, PnpSolverConfig(motion_only_ba_enabled=False))
        visual_features = make_visual_features(feat_ids, object_points, left_uv)

        def raise_cv2_error(**_kwargs: object) -> tuple[bool, None, None, None]:
            raise cv2.error("bad geometry")

        monkeypatch.setattr("core.pose_tracker.pnp_solver.cv2.solvePnPRansac", raise_cv2_error)

        result = solver.solve_visual_features(visual_features)

        assert result.status is PnpSolveStatus.PNP_FAILED
        assert not result.ok
        np.testing.assert_array_equal(result.inlier_mask, np.zeros(feat_ids.shape[0], dtype=bool))
        np.testing.assert_array_equal(result.outlier_feat_ids, feat_ids)

    def test_motion_only_ba_receives_only_inlier_correspondences(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Motion-only BA should receive only RANSAC inlier correspondences."""
        stereo_ctx = make_stereo_ctx()
        solver = PnpPoseSolver(stereo_ctx, PnpSolverConfig(motion_only_ba_enabled=True))
        object_points = make_reference_points()[:4]
        left_uv = project(SE3.identity(), object_points, stereo_ctx.stereo_k)
        right_uv = left_uv - np.array([10.0, 0.0], dtype=np.float64)
        feat_ids = np.array([10, 20, 30, 40], dtype=np.int32)
        pnp_inlier_mask = np.array([True, False, True, True])
        ba_calls = []

        def fake_pnp(_visual_features: np.ndarray) -> _PnPRansacResult:
            return _PnPRansacResult(
                status=PnpSolveStatus.SUCCESS,
                reason="fake pnp success",
                pose=SE3.identity(),
                inlier_mask=pnp_inlier_mask,
                outlier_mask=np.logical_not(pnp_inlier_mask),
            )

        def fake_ba(_pose: SE3, visual_features: np.ndarray) -> _MotionOnlyBaResult:
            ba_calls.append(visual_features)
            return _MotionOnlyBaResult(pose=_pose, ok=True, reason=None)

        monkeypatch.setattr(solver, "_estimate_cam0_in_reference_pnp_ransac", fake_pnp)
        monkeypatch.setattr(solver, "_motion_only_ba", fake_ba)
        visual_features = make_visual_features(feat_ids, object_points, left_uv, right_uv)

        result = solver.solve_visual_features(visual_features)

        assert result.ok
        assert len(ba_calls) == 1
        np.testing.assert_array_equal(ba_calls[0][:, 0], np.array([10, 30, 40], dtype=np.float64))
        np.testing.assert_array_equal(result.outlier_feat_ids, np.array([20], dtype=np.int32))
