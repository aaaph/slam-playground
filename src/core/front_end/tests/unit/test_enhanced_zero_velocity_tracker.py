import numpy as np

from core.feature_tracker.zero_velocity_tracker import ZeroVelocityTrackerState
from core.front_end.enhanced_zero_velocity_tracker import EnhancedZeroVelocityTracker
from core.pose_tracker.inertial_integration import ImuBatch, ImuBuffer, ImuSchema


def test_tracker_requires_visual_and_imu_stationarity() -> None:
    tracker = EnhancedZeroVelocityTracker()
    gyro_bias = np.array([0.08, 0.0, 0.0])
    static_imu = _imu_batch(accel=np.array([0.0, 0.0, 9.81]), gyro=gyro_bias)
    rotating_imu = _imu_batch(accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.2, 0.0, 0.0]))
    vibration = np.tile([-1.0, 1.0], 5)
    vibrating_imu = _imu_samples(
        accel=np.column_stack((vibration, np.zeros(10), np.full(10, 9.81))),
        gyro=gyro_bias + np.column_stack((0.2 * vibration, np.zeros(10), np.zeros(10))),
    )
    empty_imu = ImuBatch(np.empty((0, ImuSchema.count()), dtype=np.float64))

    assert (
        tracker.track(ZeroVelocityTrackerState.ZERO_VELOCITY, static_imu, gyro_bias=gyro_bias)
        == ZeroVelocityTrackerState.NON_ZERO_VELOCITY
    )
    assert (
        tracker.track(ZeroVelocityTrackerState.ZERO_VELOCITY, static_imu, gyro_bias=gyro_bias)
        == ZeroVelocityTrackerState.ZERO_VELOCITY
    )
    assert (
        tracker.track(ZeroVelocityTrackerState.ZERO_VELOCITY, static_imu)
        == ZeroVelocityTrackerState.NON_ZERO_VELOCITY
    )
    assert (
        tracker.track(ZeroVelocityTrackerState.ZERO_VELOCITY, rotating_imu)
        == ZeroVelocityTrackerState.NON_ZERO_VELOCITY
    )
    assert (
        tracker.track(ZeroVelocityTrackerState.ZERO_VELOCITY, static_imu, gyro_bias=gyro_bias)
        == ZeroVelocityTrackerState.NON_ZERO_VELOCITY
    )
    vibration_tracker = EnhancedZeroVelocityTracker()
    assert (
        vibration_tracker.track(ZeroVelocityTrackerState.ZERO_VELOCITY, vibrating_imu, gyro_bias=gyro_bias)
        == ZeroVelocityTrackerState.NON_ZERO_VELOCITY
    )
    assert (
        vibration_tracker.track(ZeroVelocityTrackerState.ZERO_VELOCITY, vibrating_imu, gyro_bias=gyro_bias)
        == ZeroVelocityTrackerState.ZERO_VELOCITY
    )
    primed_tracker = EnhancedZeroVelocityTracker()
    primed_tracker.confirm_stationary()
    assert (
        primed_tracker.track(ZeroVelocityTrackerState.ZERO_VELOCITY, static_imu, gyro_bias=gyro_bias)
        == ZeroVelocityTrackerState.ZERO_VELOCITY
    )
    assert tracker.track(ZeroVelocityTrackerState.ZERO_VELOCITY, empty_imu) == ZeroVelocityTrackerState.UNKNOWN
    assert tracker.track(ZeroVelocityTrackerState.UNKNOWN, static_imu) == ZeroVelocityTrackerState.UNKNOWN
    assert (
        tracker.track(ZeroVelocityTrackerState.NON_ZERO_VELOCITY, static_imu)
        == ZeroVelocityTrackerState.NON_ZERO_VELOCITY
    )


def _imu_batch(accel: np.ndarray, gyro: np.ndarray) -> ImuBatch:
    sample_count = 10
    return _imu_samples(np.tile(accel, (sample_count, 1)), np.tile(gyro, (sample_count, 1)))


def _imu_samples(accel: np.ndarray, gyro: np.ndarray) -> ImuBatch:
    sample_count = accel.shape[0]
    buffer = ImuBuffer(capacity=sample_count)
    buffer.add_batch(
        accel_batch=accel,
        gyro_batch=gyro,
        timestamp_batch=np.arange(sample_count, dtype=np.float64) * 5_000_000,
    )
    return buffer.get_last_batch()
