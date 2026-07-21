import numpy as np
from scipy.spatial.transform import Rotation

from core.feature_tracker.feature_metrics_schema import FeatureMetricsSchema, FeatureTrackerMetrics
from core.front_end.front_end_bootstrap import FrontEndBootstrap, FrontEndBootstrapDecision
from core.pose_tracker.inertial_integration import ImuBatch, ImuBuffer


class TestFrontEndBootstrap:
    """Unit tests for frontend bootstrap."""

    def test_feed_returns_unknown_and_initial_rotation_for_first_imu_batch(self) -> None:
        """Bootstrap should emit a one-shot initial rotation from the first IMU batch."""
        bootstrap = FrontEndBootstrap()
        imu_batch = self._imu_batch(accel=np.array([[0.0, 0.0, 9.81]]))

        result = bootstrap.feed(
            frame_id=0,
            _timestamp_ns=1_000_000_000.0,
            visual_metrics=self._visual_metrics(),
            imu_batch=imu_batch,
        )

        assert result.decision == FrontEndBootstrapDecision.UNKNOWN
        assert not result.ready
        assert result.rotation_ready
        np.testing.assert_allclose(result.rotation_quat, Rotation.identity().as_quat(), atol=1e-12)

    def test_feed_emits_initial_rotation_only_once(self) -> None:
        """Bootstrap should not reinitialize rotation after the first valid IMU batch."""
        bootstrap = FrontEndBootstrap()

        first_result = bootstrap.feed(
            frame_id=0,
            _timestamp_ns=1_000_000_000.0,
            visual_metrics=self._visual_metrics(),
            imu_batch=self._imu_batch(accel=np.array([[0.0, 0.0, 9.81]])),
        )
        second_result = bootstrap.feed(
            frame_id=1,
            _timestamp_ns=1_050_000_000.0,
            visual_metrics=self._visual_metrics(),
            imu_batch=self._imu_batch(accel=np.array([[9.81, 0.0, 0.0]])),
        )

        assert first_result.rotation_ready
        assert not second_result.rotation_ready
        assert second_result.decision == FrontEndBootstrapDecision.UNKNOWN

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
    def _imu_batch(*, accel: np.ndarray) -> ImuBatch:
        gyro = np.zeros_like(accel)
        timestamps = np.arange(accel.shape[0], dtype=np.float64) * 5_000_000.0
        buffer = ImuBuffer(capacity=max(accel.shape[0], 1))
        if accel.shape[0] == 0:
            return buffer.get_last_batch()
        buffer.add_batch(accel, gyro, timestamps)
        return buffer.get_last_batch()
