import numpy as np
import pytest

from core.feature_tracker.zero_velocity_tracker import ZeroVelocityTrackerState
from core.front_end.front_end_bootstrap import FrontEndBootstrap, FrontEndBootstrapDecision
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
                zero_velocity_state=ZeroVelocityTrackerState.NON_ZERO_VELOCITY,
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
                zero_velocity_state=ZeroVelocityTrackerState.NON_ZERO_VELOCITY,
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

    def test_gyro_bias_uses_only_gyro_columns_from_all_imu_rows(self) -> None:
        """Gyro bias should be a three-axis mean over committed and local samples."""
        bootstrap = FrontEndBootstrap(sample_stride=2)
        gyro = np.array([0.1, -0.2, 0.3])

        for frame_id in range(2):
            timestamp_ns = 1_000_000_000 + frame_id * 50_000_000
            bootstrap.feed(
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
                stereo_frame=self._stereo_frame(np.array([frame_id])),
                zero_velocity_state=ZeroVelocityTrackerState.NON_ZERO_VELOCITY,
                imu_batch=self._imu_batch(timestamp_ns, gyro=gyro),
            )

        np.testing.assert_allclose(bootstrap.gyro_bias_from_imu(), gyro)

    def test_evaluate_returns_unknown_until_static_window_is_ready(self) -> None:
        """Evaluation should emit rough rotation while the static decision is pending."""
        bootstrap = FrontEndBootstrap(sample_stride=1, mininal_window_size=2)
        bootstrap.feed(
            frame_id=0,
            timestamp_ns=1_000_000_000,
            stereo_frame=self._stereo_frame(np.array([0])),
            zero_velocity_state=ZeroVelocityTrackerState.ZERO_VELOCITY,
            imu_batch=self._imu_batch(1_000_000_000),
        )

        result = bootstrap.evaluate()

        assert result.decision == FrontEndBootstrapDecision.UNKNOWN
        assert result.initial_rotation is not None
        np.testing.assert_allclose(result.initial_rotation.as_quat(), [0.0, 0.0, 0.0, 1.0])
        assert result.gyro_bias is None

    def test_evaluate_recomputes_rotation_when_static_is_confirmed(self) -> None:
        """Static confirmation should replace the rough rotation using all buffered IMU samples."""
        bootstrap = FrontEndBootstrap(sample_stride=1, mininal_window_size=2)
        gyro = np.array([0.1, -0.2, 0.3])
        accel = np.array([9.81, 0.0, 0.0])

        for frame_id in range(2):
            timestamp_ns = 1_000_000_000 + frame_id * 50_000_000
            bootstrap.feed(
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
                stereo_frame=self._stereo_frame(np.array([frame_id])),
                zero_velocity_state=ZeroVelocityTrackerState.ZERO_VELOCITY,
                imu_batch=self._imu_batch(timestamp_ns, gyro=gyro, accel=accel if frame_id else None),
            )
            if frame_id == 0:
                rough_result = bootstrap.evaluate()

        static_result = bootstrap.evaluate()
        repeated_result = bootstrap.evaluate()

        assert rough_result.decision == FrontEndBootstrapDecision.UNKNOWN
        assert rough_result.initial_rotation is not None
        np.testing.assert_allclose(rough_result.initial_rotation.as_quat(), [0.0, 0.0, 0.0, 1.0])
        assert rough_result.gyro_bias is None
        assert static_result.decision == FrontEndBootstrapDecision.STATIC
        assert static_result.initial_rotation is not None
        accel_mean = np.array([4.905, 0.0, 4.905])
        np.testing.assert_allclose(
            static_result.initial_rotation.apply(accel_mean),
            [0.0, 0.0, np.linalg.norm(accel_mean)],
            atol=1e-12,
        )
        assert static_result.gyro_bias is not None
        np.testing.assert_allclose(static_result.gyro_bias, gyro)
        assert repeated_result.decision == FrontEndBootstrapDecision.STATIC
        assert repeated_result.initial_rotation is not None
        np.testing.assert_allclose(
            repeated_result.initial_rotation.as_quat(), static_result.initial_rotation.as_quat()
        )
        assert repeated_result.gyro_bias is not None
        np.testing.assert_allclose(repeated_result.gyro_bias, gyro)

    @staticmethod
    def _stereo_frame(feature_ids: np.ndarray) -> np.ndarray:
        frame = np.zeros((feature_ids.shape[0], StereoTriangulationSchema.count()), dtype=np.float32)
        frame[:, StereoTriangulationSchema.FEAT_ID] = feature_ids
        return frame

    @staticmethod
    def _imu_batch(
        timestamp_ns: int, *, gyro: np.ndarray | None = None, accel: np.ndarray | None = None
    ) -> ImuBatch:
        buffer = ImuBuffer(capacity=1)
        buffer.add_batch(
            accel_batch=np.array([[0.0, 0.0, 9.81]]) if accel is None else accel[None, :],
            gyro_batch=np.zeros((1, 3)) if gyro is None else gyro[None, :],
            timestamp_batch=np.array([timestamp_ns]),
        )
        return buffer.get_last_batch()
