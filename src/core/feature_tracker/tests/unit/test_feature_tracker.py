from typing import cast

import cv2
import numpy as np
import pytest

from core.camera_model.stereo_camera_ctx import StereoContext
from core.feature_tracker.feature_frame import FeatureFrame
from core.feature_tracker.feature_metrics_schema import FeatureMetricsSchema
from core.feature_tracker.feature_schema import FeatureLifecycle, FeatureSchema
from core.feature_tracker.feature_tracker import FeatureTracker, StereoMatchSchema
from core.feature_tracker.zero_velocity_tracker import ZeroVelocityTrackerState
from core.transformations.special_euclidian_3_dim import SE3


class TestFeatureTracker:
    """Unit test for feature tracker."""

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
    def feature_tracker(self, stereo_ctx: StereoContext) -> FeatureTracker:
        """Create a feature tracker."""
        return FeatureTracker.default_factory(stereo_ctx)

    def test_feature_tracker_get_active_features_ids(self, feature_tracker: FeatureTracker):
        """Test that the feature tracker has a get_active_features_ids method."""
        for i in range(10):
            feature_tracker.tensor.add(i, 1, (0, 0), (1, 1), FeatureLifecycle.ACTIVE)
        active_features_ids = feature_tracker.active_frame().ids
        assert len(active_features_ids) == 10
        for i in range(10):
            assert i in active_features_ids

    def test_feature_tracker_metrics_include_zero_velocity_state(self, feature_tracker: FeatureTracker):
        """Tracker metrics should expose debounced zero-velocity state."""
        data = np.full((4, FeatureSchema.count()), np.nan, dtype=np.float32)
        data[:, FeatureSchema.FEAT_ID] = [1, 2, 3, 4]
        data[:, FeatureSchema.TIMESTAMP] = 1
        data[:, FeatureSchema.LEFT_U : FeatureSchema.LEFT_V + 1] = [
            [10, 10],
            [20, 20],
            [30, 30],
            [40, 40],
        ]
        data[:, FeatureSchema.RIGHT_U : FeatureSchema.RIGHT_V + 1] = [
            [9, 10],
            [19, 20],
            [np.nan, np.nan],
            [39, 40],
        ]
        data[:, FeatureSchema.LIFECYCLE] = [
            FeatureLifecycle.ACTIVE.value,
            FeatureLifecycle.ACTIVE.value,
            FeatureLifecycle.ACTIVE.value,
            FeatureLifecycle.LOST.value,
        ]
        data[:, FeatureSchema.AGE] = [0, 1, 2, 3]
        tracking_mask = data[:, FeatureSchema.LIFECYCLE] == FeatureLifecycle.ACTIVE.value

        metrics = feature_tracker.metrics
        feature_tracker.temporal_pixel_displacement = 0.0
        feature_tracker.temporal_pixel_displacement_p90 = 0.0
        for _ in range(4):
            feature_tracker._update_metrics(tracking_mask, data)  # noqa: SLF001

        assert feature_tracker.metrics is metrics
        assert feature_tracker.metrics.ndarray is feature_tracker.metrics_array
        assert feature_tracker.metrics.good_count == 3
        assert feature_tracker.metrics.lost_count == 1
        assert feature_tracker.metrics.tracked_count == 2
        assert feature_tracker.metrics.stereo_ok_count == 2
        assert feature_tracker.metrics.stereo_ok_ratio == pytest.approx(2 / 3)
        assert feature_tracker.metrics.temporal_pixel_displacement_p90 == 0.0
        assert feature_tracker.metrics.zero_velocity_state == ZeroVelocityTrackerState.ZERO_VELOCITY
        assert feature_tracker.metrics_array[FeatureMetricsSchema.GOOD_COUNT] == 3

        feature_tracker.temporal_pixel_displacement = 2.0
        feature_tracker.temporal_pixel_displacement_p90 = 7.0
        for _ in range(4):
            feature_tracker._update_metrics(tracking_mask, data)  # noqa: SLF001

        assert feature_tracker.metrics is metrics
        assert feature_tracker.metrics.temporal_pixel_displacement == 2.0
        assert feature_tracker.metrics.temporal_pixel_displacement_p90 == 7.0
        assert feature_tracker.metrics.zero_velocity_state == ZeroVelocityTrackerState.NON_ZERO_VELOCITY
        assert feature_tracker.metrics_array[FeatureMetricsSchema.TEMPORAL_PIXEL_DISPLACEMENT_P90] == 7.0
        assert (
            feature_tracker.metrics_array[FeatureMetricsSchema.ZERO_VELOCITY_STATE]
            == ZeroVelocityTrackerState.NON_ZERO_VELOCITY.value
        )

    def test_optical_flow_rejects_large_displacement_after_forward_backward_check(
        self,
        feature_tracker: FeatureTracker,
        monkeypatch,
    ):
        """Test that a huge LK jump is rejected even if backward LK returns to the source point."""
        feature_tracker.left_prev = np.zeros((480, 752), dtype=np.uint8)
        left_next = np.zeros_like(feature_tracker.left_prev)
        prev_data = np.array(
            [
                [1, 1, 100, 40, np.nan, np.nan, FeatureLifecycle.ACTIVE.value, 0],
                [2, 1, 100, 80, np.nan, np.nan, FeatureLifecycle.ACTIVE.value, 0],
            ],
            dtype=np.float32,
        )
        prev_frame = FeatureFrame(
            data=prev_data,
            active_indeces=np.array([0, 1], dtype=np.int32),
            active_mask=np.array([True, True]),
            timestamp=1,
        )

        calls = 0

        def fake_lk(_prev_img, _next_img, _points, _next_points, **_params):
            nonlocal calls
            calls += 1
            status = np.ones((2, 1), dtype=np.uint8)
            err = np.zeros((2, 1), dtype=np.float32)
            if calls == 1:
                return np.array([[110, 40], [400, 80]], dtype=np.float32), status, err
            return np.array([[100, 40], [100, 80]], dtype=np.float32), status, err

        monkeypatch.setattr("core.feature_tracker.feature_tracker.cv2.calcOpticalFlowPyrLK", fake_lk)

        result = feature_tracker._optical_flow_lk(left_next, prev_frame)  # noqa: SLF001

        np.testing.assert_allclose(result[0, FeatureSchema.LEFT_U : FeatureSchema.LEFT_V + 1], [110, 40])
        np.testing.assert_allclose(result[1, FeatureSchema.LEFT_U : FeatureSchema.LEFT_V + 1], [100, 80])
        assert result[0, FeatureSchema.LIFECYCLE] == FeatureLifecycle.ACTIVE.value
        assert result[1, FeatureSchema.LIFECYCLE] == FeatureLifecycle.LOST.value
        assert np.all(result[:, FeatureSchema.AGE] == 1)
        assert feature_tracker.temporal_pixel_displacement == 10.0
        assert feature_tracker.temporal_pixel_displacement_p90 == 10.0

    def test_optical_flow_uses_original_points_when_lk_mutates_inputs(
        self,
        feature_tracker: FeatureTracker,
        monkeypatch,
    ):
        """Test that LK input mutation cannot hide a large temporal jump."""
        feature_tracker.left_prev = np.zeros((480, 752), dtype=np.uint8)
        left_next = np.zeros_like(feature_tracker.left_prev)
        prev_data = np.array(
            [
                [1, 1, 100, 40, np.nan, np.nan, FeatureLifecycle.ACTIVE.value, 0],
                [2, 1, 100, 80, np.nan, np.nan, FeatureLifecycle.ACTIVE.value, 0],
            ],
            dtype=np.float32,
        )
        prev_frame = FeatureFrame(
            data=prev_data,
            active_indeces=np.array([0, 1], dtype=np.int32),
            active_mask=np.array([True, True]),
            timestamp=1,
        )

        calls = 0

        def fake_lk(_prev_img, _next_img, points, next_points, **_params):
            nonlocal calls
            calls += 1
            status = np.ones((2, 1), dtype=np.uint8)
            err = np.zeros((2, 1), dtype=np.float32)
            if calls == 1:
                tracked_points = np.array([[110, 40], [400, 80]], dtype=np.float32)
                points[:] = tracked_points
                if next_points is not None:
                    next_points[:] = tracked_points
                return tracked_points, status, err
            return next_points.copy(), status, err

        monkeypatch.setattr("core.feature_tracker.feature_tracker.cv2.calcOpticalFlowPyrLK", fake_lk)

        result = feature_tracker._optical_flow_lk(left_next, prev_frame)  # noqa: SLF001

        assert result[0, FeatureSchema.LIFECYCLE] == FeatureLifecycle.ACTIVE.value
        assert result[1, FeatureSchema.LIFECYCLE] == FeatureLifecycle.LOST.value

    def test_stereo_match_lk_returns_match_and_mask_columns(self, feature_tracker: FeatureTracker, monkeypatch):
        """Stereo LK result should keep one row per input point and carry stereo_ok."""
        left = np.zeros((480, 752), dtype=np.uint8)
        right = np.zeros_like(left)
        points_left = np.array(
            [
                [1, 100, 10],
                [2, 100, 20],
                [3, 100, 30],
            ],
            dtype=np.float32,
        )
        calls = 0

        def fake_lk(_prev_img, _next_img, _points, _next_points, **_params):
            nonlocal calls
            calls += 1
            status = np.ones((3, 1), dtype=np.uint8)
            err = np.zeros((3, 1), dtype=np.float32)
            if calls == 1:
                return np.array([[90, 10], [105, 20], [70, 30]], dtype=np.float32), status, err
            return points_left[:, 1:].copy(), status, err

        monkeypatch.setattr("core.feature_tracker.feature_tracker.cv2.calcOpticalFlowPyrLK", fake_lk)

        stereo_match = feature_tracker._stereo_match_lk(left, right, points_left)  # noqa: SLF001

        assert stereo_match.shape == (3, StereoMatchSchema.count())
        np.testing.assert_array_equal(stereo_match[:, StereoMatchSchema.STEREO_OK], np.array([1, 0, 1]))
        np.testing.assert_allclose(
            stereo_match[:, StereoMatchSchema.RIGHT_U : StereoMatchSchema.RIGHT_V + 1],
            np.array([[90, 10], [np.nan, np.nan], [70, 30]], dtype=np.float32),
            equal_nan=True,
        )

    def test_stereo_match_lk_allows_small_rectification_residual(
        self, feature_tracker: FeatureTracker, monkeypatch
    ):
        """Stereo LK should tolerate small vertical residuals after rectification."""
        left = np.zeros((480, 752), dtype=np.uint8)
        right = np.zeros_like(left)
        points_left = np.array(
            [
                [1, 100, 10],
                [2, 100, 20],
            ],
            dtype=np.float32,
        )
        calls = 0

        def fake_lk(_prev_img, _next_img, _points, _next_points, **_params):
            nonlocal calls
            calls += 1
            status = np.ones((2, 1), dtype=np.uint8)
            err = np.zeros((2, 1), dtype=np.float32)
            if calls == 1:
                return np.array([[90, 12], [90, 23]], dtype=np.float32), status, err
            return points_left[:, 1:].copy(), status, err

        monkeypatch.setattr("core.feature_tracker.feature_tracker.cv2.calcOpticalFlowPyrLK", fake_lk)

        stereo_match = feature_tracker._stereo_match_lk(left, right, points_left)  # noqa: SLF001

        np.testing.assert_array_equal(stereo_match[:, StereoMatchSchema.STEREO_OK], np.array([1, 0]))
        np.testing.assert_allclose(
            stereo_match[:, StereoMatchSchema.RIGHT_U : StereoMatchSchema.RIGHT_V + 1],
            np.array([[90, 10], [np.nan, np.nan]], dtype=np.float32),
            equal_nan=True,
        )

    def test_feed_tracked_stereo_score_uses_stereo_match_ok_column(
        self, feature_tracker: FeatureTracker, monkeypatch
    ):
        """Feed should increment tracked stereo score only for stereo_ok rows."""
        batch = np.full((4, FeatureSchema.count()), np.nan, dtype=np.float32)
        batch[:, FeatureSchema.FEAT_ID] = [1, 2, 3, 4]
        batch[:, FeatureSchema.LEFT_U : FeatureSchema.LEFT_V + 1] = [
            [30, 31],
            [40, 41],
            [50, 51],
            [60, 61],
        ]
        batch[:, FeatureSchema.LIFECYCLE] = [
            FeatureLifecycle.ACTIVE.value,
            FeatureLifecycle.ACTIVE.value,
            FeatureLifecycle.LOST.value,
            FeatureLifecycle.ACTIVE.value,
        ]
        batch[:, FeatureSchema.STEREO_SCORE] = [2, 3, 4, 5]
        batch[:, FeatureSchema.TIMESTAMP] = 1
        feature_tracker.tensor.add_batch(1, batch)
        stereo_match = np.array(
            [
                [1, 30, 31, 29, 31, 1],
                [2, 40, 41, np.nan, np.nan, 0],
                [4, 60, 61, 59, 61, 1],
            ],
            dtype=np.float32,
        )
        next_batch = batch.copy()
        next_batch[2, FeatureSchema.LIFECYCLE] = FeatureLifecycle.LOST.value

        class NoKeypoints:
            def detect(self, **_kwargs):
                return []

        monkeypatch.setattr(feature_tracker, "_optical_flow_lk", lambda *_args: next_batch)
        monkeypatch.setattr(feature_tracker, "_stereo_match_lk", lambda *_args: stereo_match)
        feature_tracker.fast = cast("cv2.FastFeatureDetector", NoKeypoints())

        left = np.zeros((100, 100), dtype=np.uint8)
        right = np.zeros((100, 100), dtype=np.uint8)
        tracking_mask, tracking_frame = feature_tracker.feed(2, (left, right))
        rows_by_id = {int(row[FeatureSchema.FEAT_ID]): row for row in tracking_frame}

        np.testing.assert_allclose(
            [rows_by_id[feat_id][FeatureSchema.STEREO_SCORE] for feat_id in (1, 2, 3, 4)],
            np.array([3, 0, 0, 6], dtype=np.float32),
        )
        np.testing.assert_allclose(
            np.vstack(
                [
                    rows_by_id[feat_id][FeatureSchema.RIGHT_U : FeatureSchema.RIGHT_V + 1]
                    for feat_id in (1, 2, 3, 4)
                ]
            ),
            np.array([[29, 31], [np.nan, np.nan], [np.nan, np.nan], [59, 61]], dtype=np.float32),
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            tracking_mask,
            tracking_frame[:, FeatureSchema.LIFECYCLE] == FeatureLifecycle.ACTIVE.value,
        )

    def test_initiate_new_features_passes_ndarray_to_stereo_match(
        self, feature_tracker: FeatureTracker, monkeypatch
    ):
        """New OpenCV keypoints should be converted to Nx3 rows before stereo matching."""
        keypoints = [
            cv2.KeyPoint(10, 11, 1, response=0.3),
            cv2.KeyPoint(20, 21, 1, response=0.8),
        ]
        captured_points = []

        def fake_stereo_match(_left, _right, points_left):
            captured_points.append(points_left.copy())
            result = np.full((points_left.shape[0], StereoMatchSchema.count()), np.nan, dtype=np.float32)
            result[:, StereoMatchSchema.FEAT_ID : StereoMatchSchema.LEFT_V + 1] = points_left
            result[:, StereoMatchSchema.RIGHT_U : StereoMatchSchema.RIGHT_V + 1] = points_left[:, 1:3] - [1, 0]
            result[:, StereoMatchSchema.STEREO_OK] = 1.0
            return result

        monkeypatch.setattr(feature_tracker, "_stereo_match_lk", fake_stereo_match)

        batch = feature_tracker.initiate_new_features(
            np.zeros((10, 10), dtype=np.uint8),
            np.zeros((10, 10), dtype=np.uint8),
            keypoints,
            timestamp=1,
        )

        assert len(captured_points) == 1
        assert isinstance(captured_points[0], np.ndarray)
        np.testing.assert_allclose(captured_points[0][:, 1:3], np.array([[20, 21], [10, 11]], dtype=np.float32))
        np.testing.assert_allclose(
            batch[:, FeatureSchema.STEREO_SCORE],
            np.zeros(batch.shape[0], dtype=np.float32),
        )

    def test_initiate_new_features_does_not_apply_global_retrack_cap(
        self, feature_tracker: FeatureTracker, monkeypatch
    ):
        """Retrack quota is handled before feature rows are created."""
        keypoints = [cv2.KeyPoint(float(10 + i), 11, 1, response=float(i)) for i in range(12)]

        def fake_stereo_match(_left, _right, points_left):
            result = np.full((points_left.shape[0], StereoMatchSchema.count()), np.nan, dtype=np.float32)
            result[:, StereoMatchSchema.FEAT_ID : StereoMatchSchema.LEFT_V + 1] = points_left
            result[:, StereoMatchSchema.RIGHT_U : StereoMatchSchema.RIGHT_V + 1] = points_left[:, 1:3] - [1, 0]
            result[:, StereoMatchSchema.STEREO_OK] = 1.0
            return result

        monkeypatch.setattr(feature_tracker, "_stereo_match_lk", fake_stereo_match)

        batch = feature_tracker.initiate_new_features(
            np.zeros((20, 40), dtype=np.uint8),
            np.zeros((20, 40), dtype=np.uint8),
            keypoints,
            timestamp=1,
        )

        assert batch.shape[0] == len(keypoints)

    def test_select_retrack_kps_respects_region_quotas(self, stereo_ctx: StereoContext):
        """One FAST pass should still distribute selected retrack points across hungry regions."""
        feature_tracker = FeatureTracker.default_factory(
            stereo_ctx,
            feat_amount_per_region=2,
            feat_retrack_threshold=1,
        )
        keypoints = [
            cv2.KeyPoint(400, 30, 1, response=1.0),  # outside target regions
            cv2.KeyPoint(200, 30, 1, response=0.95),
            cv2.KeyPoint(30, 30, 1, response=0.9),
            cv2.KeyPoint(230, 30, 1, response=0.85),
            cv2.KeyPoint(35, 30, 1, response=0.8),  # too close to the stronger region-0 keypoint
            cv2.KeyPoint(60, 30, 1, response=0.7),
        ]
        region_counts = np.zeros(feature_tracker.REGION_AMOUNT, dtype=np.int64)

        selected = feature_tracker._select_retrack_kps(  # noqa: SLF001
            keypoints,
            region_counts=region_counts,
            target_region_ids=np.array([0, 1], dtype=np.int64),
            min_distance=20,
        )

        selected_regions = [int(feature_tracker.grid_mask[int(kp.pt[1]), int(kp.pt[0])]) for kp in selected]
        assert len(selected) == 4
        assert selected_regions.count(0) == 2
        assert selected_regions.count(1) == 2
        assert 2 not in selected_regions
