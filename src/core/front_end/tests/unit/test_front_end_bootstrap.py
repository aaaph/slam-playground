import numpy as np
import pytest

from core.feature_tracker.feature_metrics_schema import FeatureMetricsSchema, FeatureTrackerMetrics
from core.front_end.front_end_bootstrap import FrontEndBootstrap
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema
from core.pose_tracker.inertial_integration import ImuBatch, ImuBuffer


class TestFrontEndBootstrap:
    """Unit tests for frontend bootstrap."""

    def test_feed_stores_sampled_frames_with_aggregated_imu(self) -> None:
        """A stored frame should include every IMU batch since the previous stored frame."""
        bootstrap = FrontEndBootstrap(sample_stride=2)

        for frame_id in range(3):
            timestamp_ns = 1_000_000_000 + frame_id * 50_000_000
            bootstrap.feed(
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
                stereo_frame=self._stereo_frame(np.array([frame_id])),
                visual_metrics=self._visual_metrics(),
                imu_batch=self._imu_batch(timestamp_ns),
            )

        np.testing.assert_array_equal(bootstrap.sliding_window.frame_ids, [0, 2])
        np.testing.assert_array_equal(bootstrap.sliding_window.imu_counts, [1, 2])
        np.testing.assert_array_equal(
            bootstrap.sliding_window.imu[1, :2, 0],
            [1_050_000_000, 1_100_000_000],
        )

    def test_sliding_window_wraps_in_chronological_order_and_clears_stale_rows(self) -> None:
        """A wrapped window should expose only current rows in chronological order."""
        bootstrap = FrontEndBootstrap(sample_stride=1)

        for frame_id in range(11):
            feature_ids = np.array([frame_id, 100]) if frame_id == 0 else np.array([frame_id])
            timestamp_ns = 1_000_000_000 + frame_id * 50_000_000
            bootstrap.feed(
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
                stereo_frame=self._stereo_frame(feature_ids),
                visual_metrics=self._visual_metrics(),
                imu_batch=self._imu_batch(timestamp_ns),
            )

        window = bootstrap.sliding_window
        assert window.size == 10
        np.testing.assert_array_equal(window.frame_ids, np.arange(1, 11))
        np.testing.assert_array_equal(window.feature_counts, np.ones(10))
        assert np.isnan(window.features[-1, 1:]).all()

    def test_sample_stride_must_be_positive(self) -> None:
        """A non-positive sampling stride is invalid."""
        with pytest.raises(ValueError, match="sample_stride must be at least 1"):
            FrontEndBootstrap(sample_stride=0)

    @staticmethod
    def _stereo_frame(feature_ids: np.ndarray) -> np.ndarray:
        frame = np.zeros((feature_ids.shape[0], StereoTriangulationSchema.count()), dtype=np.float32)
        frame[:, StereoTriangulationSchema.FEAT_ID] = feature_ids
        return frame

    @staticmethod
    def _visual_metrics() -> FeatureTrackerMetrics:
        metrics = np.zeros(FeatureMetricsSchema.count(), dtype=np.float32)
        metrics[FeatureMetricsSchema.ACTIVE_COUNT] = 80
        metrics[FeatureMetricsSchema.GOOD_COUNT] = 80
        metrics[FeatureMetricsSchema.TRACKED_COUNT] = 80
        metrics[FeatureMetricsSchema.STEREO_OK_COUNT] = 80
        metrics[FeatureMetricsSchema.STEREO_OK_RATIO] = 1.0
        return FeatureTrackerMetrics(metrics)

    @staticmethod
    def _imu_batch(timestamp_ns: int) -> ImuBatch:
        buffer = ImuBuffer(capacity=1)
        buffer.add_batch(
            accel_batch=np.array([[0.0, 0.0, 9.81]]),
            gyro_batch=np.zeros((1, 3)),
            timestamp_batch=np.array([timestamp_ns]),
        )
        return buffer.get_last_batch()
