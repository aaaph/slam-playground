from typing import cast

import numpy as np

from core.camera_model.stereo_camera_ctx import StereoContext
from core.pose_tracker.frame_to_frame_pnp_estimator import FrameToFramePnPEstimator
from core.pose_tracker.frame_to_frame_pnp_store import FrameToFramePnpStore, PnPMapSchema
from core.pose_tracker.pnp_solver import PnpPoseResult, PnpPoseSolver, PnpSolveStatus
from core.transformations.special_euclidian_3_dim import SE3


class FakePnpSolver:
    """Fake PnP solver that records the estimator input."""

    def __init__(self, result: PnpPoseResult | None = None) -> None:
        """Initialize the fake solver."""
        self.result = result
        self.calls: list[np.ndarray] = []

    def solve_visual_features(self, visual_features: np.ndarray) -> PnpPoseResult:
        """Record visual features and return the configured result."""
        self.calls.append(visual_features.copy())
        if self.result is None:
            raise AssertionError("FakePnpSolver.result must be configured before solve")
        return self.result


def make_stereo_ctx(cam0_in_body: SE3 | None = None) -> StereoContext:
    """Create a stereo context for estimator tests."""
    k_matrix = np.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]])
    cam0_in_body = cam0_in_body or SE3.identity()
    return StereoContext(
        resolution=(640, 480),
        stereo_k=k_matrix,
        cam0_k=k_matrix,
        cam1_k=k_matrix,
        baseline=0.1,
        cam0_in_body_se3=cam0_in_body,
        cam1_in_body_se3=SE3.identity(),
    )


def make_pnp_rows(feat_ids: np.ndarray, *, xyz_offset: float = 0.0, uv_offset: float = 100.0) -> np.ndarray:
    """Create PnP rows with distinguishable XYZ and image coordinates."""
    feat_ids = np.asarray(feat_ids, dtype=np.float64)
    rows = np.full((feat_ids.shape[0], PnPMapSchema.count()), np.nan, dtype=np.float64)
    rows[:, PnPMapSchema.FEAT_ID] = feat_ids
    rows[:, PnPMapSchema.XYZ] = np.column_stack(
        (
            feat_ids + xyz_offset,
            feat_ids + xyz_offset + 0.1,
            feat_ids + xyz_offset + 0.2,
        )
    )
    rows[:, PnPMapSchema.LEFT_UV] = np.column_stack((feat_ids + uv_offset, feat_ids + uv_offset + 1.0))
    rows[:, PnPMapSchema.RIGHT_UV] = rows[:, PnPMapSchema.LEFT_UV] - np.array([5.0, 0.0])
    return rows


def make_success_result(cam0_in_reference: SE3, inlier_count: int) -> PnpPoseResult:
    """Create a successful solver result."""
    return PnpPoseResult(
        status=PnpSolveStatus.SUCCESS,
        reason="fake success",
        cam0_in_reference=cam0_in_reference,
        inlier_feat_ids=np.arange(inlier_count, dtype=np.int32),
        outlier_feat_ids=np.empty((0,), dtype=np.int32),
        inlier_mask=np.ones((inlier_count,), dtype=np.bool_),
        reprojection_errors=np.zeros((inlier_count,), dtype=np.float64),
    )


def make_success_result_with_inlier_mask(cam0_in_reference: SE3, inlier_mask: np.ndarray) -> PnpPoseResult:
    """Create a successful solver result with the provided inlier mask."""
    feat_ids = np.arange(inlier_mask.shape[0], dtype=np.int32)
    return PnpPoseResult(
        status=PnpSolveStatus.SUCCESS,
        reason="fake success",
        cam0_in_reference=cam0_in_reference,
        inlier_feat_ids=feat_ids[inlier_mask],
        outlier_feat_ids=feat_ids[~inlier_mask],
        inlier_mask=inlier_mask,
        reprojection_errors=np.zeros(inlier_mask.shape[0], dtype=np.float64),
    )


def assert_pose_allclose(actual: SE3, expected: SE3) -> None:
    """Assert two SE3 poses are numerically equal."""
    np.testing.assert_allclose(actual.translation(), expected.translation(), atol=1e-9)
    np.testing.assert_allclose(actual.rotation().as_matrix(), expected.rotation().as_matrix(), atol=1e-9)


class TestFrameToFramePnPEstimator:
    """Unit tests for FrameToFramePnPEstimator."""

    def test_first_estimate_seeds_store_and_returns_previous_pose(self) -> None:
        """First frame should only seed the PnP store."""
        store = FrameToFramePnpStore(map_capacity=8)
        fake_solver = FakePnpSolver()
        estimator = FrameToFramePnPEstimator(store, cast("PnpPoseSolver", fake_solver), make_stereo_ctx())
        prev_pose = SE3.from_rpy_xyz(np.array([0.1, -0.2, 0.3]), np.array([1.0, 2.0, 3.0]))
        visual_features = make_pnp_rows(np.array([10, 20], dtype=np.int32))

        result = estimator.estimate_pose(prev_pose, visual_features)

        assert_pose_allclose(result, prev_pose)
        assert fake_solver.calls == []
        assert estimator.iteration == 1
        previous_mask, previous_xyz = store.get_previous_xyz(np.array([10, 20], dtype=np.int32))
        np.testing.assert_array_equal(previous_mask, np.array([True, True]))
        np.testing.assert_allclose(previous_xyz, visual_features[:, PnPMapSchema.XYZ])

    def test_estimate_pose_uses_previous_xyz_and_current_uv_for_matched_features(self) -> None:
        """Estimator should solve using previous-frame XYZ and current-frame UV."""
        cam0_in_body = SE3.from_rpy_xyz(np.array([0.02, -0.01, 0.03]), np.array([0.1, -0.2, 0.05]))
        stereo_ctx = make_stereo_ctx(cam0_in_body)
        prev_pose = SE3.from_rpy_xyz(np.array([0.1, 0.2, -0.05]), np.array([1.0, 2.0, 0.5]))
        cam0_in_reference = SE3.from_rpy_xyz(np.array([0.01, -0.03, 0.02]), np.array([0.3, -0.1, 0.2]))
        fake_solver = FakePnpSolver(make_success_result(cam0_in_reference, inlier_count=2))
        store = FrameToFramePnpStore(map_capacity=8)
        estimator = FrameToFramePnPEstimator(store, cast("PnpPoseSolver", fake_solver), stereo_ctx)
        previous_features = make_pnp_rows(np.array([10, 20, 30], dtype=np.int32), xyz_offset=100.0)
        current_features = make_pnp_rows(np.array([20, 40, 10], dtype=np.int32), xyz_offset=200.0, uv_offset=300.0)

        estimator.estimate_pose(prev_pose, previous_features)
        result = estimator.estimate_pose(prev_pose, current_features)

        assert len(fake_solver.calls) == 1
        solver_features = fake_solver.calls[0]
        np.testing.assert_array_equal(solver_features[:, PnPMapSchema.FEAT_ID], np.array([20.0, 10.0]))
        np.testing.assert_allclose(
            solver_features[:, PnPMapSchema.XYZ],
            np.vstack(
                (
                    previous_features[1, PnPMapSchema.XYZ],
                    previous_features[0, PnPMapSchema.XYZ],
                )
            ),
        )
        np.testing.assert_allclose(
            solver_features[:, PnPMapSchema.LEFT_UV],
            current_features[[0, 2], PnPMapSchema.LEFT_UV],
        )
        np.testing.assert_allclose(
            solver_features[:, PnPMapSchema.RIGHT_UV],
            current_features[[0, 2], PnPMapSchema.RIGHT_UV],
        )

        expected_pose = prev_pose * cam0_in_body * cam0_in_reference * cam0_in_body.inverse()
        assert_pose_allclose(result, expected_pose)
        assert estimator.iteration == 2

        previous_mask, previous_xyz = store.get_previous_xyz(np.array([20, 40, 10], dtype=np.int32))
        np.testing.assert_array_equal(previous_mask, np.array([True, True, True]))
        np.testing.assert_allclose(previous_xyz, current_features[:, PnPMapSchema.XYZ])

    def test_add_visual_data_seeds_previous_xyz_for_next_estimate(self) -> None:
        """Externally initialized features should be usable by the next PnP estimate."""
        stereo_ctx = make_stereo_ctx()
        prev_pose = SE3.identity()
        fake_solver = FakePnpSolver(make_success_result(SE3.identity(), inlier_count=1))
        store = FrameToFramePnpStore(map_capacity=8)
        estimator = FrameToFramePnPEstimator(store, cast("PnpPoseSolver", fake_solver), stereo_ctx)
        previous_features = make_pnp_rows(np.array([10], dtype=np.int32), xyz_offset=100.0)
        seeded_features = make_pnp_rows(np.array([20], dtype=np.int32), xyz_offset=500.0)
        current_features = make_pnp_rows(np.array([20], dtype=np.int32), xyz_offset=900.0, uv_offset=300.0)

        estimator.estimate_pose(prev_pose, previous_features)
        estimator.add_visual_data(seeded_features)
        estimator.estimate_pose(prev_pose, current_features)

        assert len(fake_solver.calls) == 1
        solver_features = fake_solver.calls[0]
        np.testing.assert_array_equal(solver_features[:, PnPMapSchema.FEAT_ID], np.array([20.0]))
        np.testing.assert_allclose(solver_features[:, PnPMapSchema.XYZ], seeded_features[:, PnPMapSchema.XYZ])
        np.testing.assert_allclose(
            solver_features[:, PnPMapSchema.LEFT_UV],
            current_features[:, PnPMapSchema.LEFT_UV],
        )

    def test_estimate_pose_updates_outlier_streak_for_matched_features_only(self) -> None:
        """Estimator should apply PnP feedback to solver input rows, not the full current frame."""
        stereo_ctx = make_stereo_ctx()
        prev_pose = SE3.identity()
        cam0_in_reference = SE3.from_rpy_xyz(np.zeros(3), np.array([0.1, 0.0, 0.0]))
        fake_solver = FakePnpSolver(
            make_success_result_with_inlier_mask(cam0_in_reference, np.array([True, False], dtype=np.bool_))
        )
        store = FrameToFramePnpStore(map_capacity=8)
        estimator = FrameToFramePnPEstimator(store, cast("PnpPoseSolver", fake_solver), stereo_ctx)
        previous_features = make_pnp_rows(np.array([10, 20, 30], dtype=np.int32), xyz_offset=100.0)
        current_features = make_pnp_rows(np.array([20, 40, 10], dtype=np.int32), xyz_offset=200.0, uv_offset=300.0)

        estimator.estimate_pose(prev_pose, previous_features)
        estimator.estimate_pose(prev_pose, current_features)

        assert store._outlier_streak[store._feat_to_slot[20]] == 0  # noqa: SLF001
        assert store._outlier_streak[store._feat_to_slot[10]] == 1  # noqa: SLF001
        assert store._outlier_streak[store._feat_to_slot[40]] == 0  # noqa: SLF001
