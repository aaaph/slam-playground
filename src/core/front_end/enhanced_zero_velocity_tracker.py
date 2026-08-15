from collections import deque
from typing import NamedTuple

import numpy as np

from core.feature_tracker.zero_velocity_tracker import ZeroVelocityTrackerState
from core.pose_tracker.inertial_integration import ImuBatch
from logger import spawn_logger


class EnhancedZeroVelocityThresholds(NamedTuple):
    """Thresholds for confirming visual zero velocity with IMU data."""

    window_size: int = 4
    min_sample_count: int = 2
    min_stationary_frames: int = 2
    gravity_magnitude: float = 9.81
    max_gyro_mean_norm: float = 0.03
    max_accel_mean_norm_error: float = 0.10


class EnhancedZeroVelocityTracker:
    """Enhanced zero velocity tracker. Module should be used to combine tracking ZUPT with imu data."""

    def __init__(self, thresholds: EnhancedZeroVelocityThresholds | None = None) -> None:
        """Initialize the enhanced zero velocity tracker."""
        self.logger = spawn_logger("ezupt")
        self.thresholds = thresholds or EnhancedZeroVelocityThresholds()
        self.imu_window: deque[np.ndarray] = deque(maxlen=self.thresholds.window_size)
        self.stationary_frame_count = 0

    def confirm_stationary(self) -> None:
        """Seed entry hysteresis from an external stationary bootstrap decision."""
        self.stationary_frame_count = self.thresholds.min_stationary_frames

    def track(
        self,
        tracking_zupt: ZeroVelocityTrackerState,
        imu_batch: ImuBatch,
        *,
        accel_bias: np.ndarray | None = None,
        gyro_bias: np.ndarray | None = None,
    ) -> ZeroVelocityTrackerState:
        """Confirm a visual zero-velocity decision with synchronized IMU data."""
        if imu_batch.sample_count == 0:
            self.imu_window.clear()
            self.stationary_frame_count = 0
            return (
                ZeroVelocityTrackerState.UNKNOWN
                if tracking_zupt == ZeroVelocityTrackerState.ZERO_VELOCITY
                else tracking_zupt
            )

        self.imu_window.append(imu_batch.rows.copy())
        if tracking_zupt != ZeroVelocityTrackerState.ZERO_VELOCITY:
            self.stationary_frame_count = 0
            return tracking_zupt

        window_batch = ImuBatch(np.concatenate(tuple(self.imu_window), axis=0))
        imu_metrics = window_batch.metrics(accel_bias=accel_bias, gyro_bias=gyro_bias)
        thresholds = self.thresholds
        if imu_metrics.sample_count < thresholds.min_sample_count:
            return ZeroVelocityTrackerState.UNKNOWN

        gyro_mean_norm = float(np.linalg.norm(imu_metrics.gyro_mean))
        accel_mean_norm_error = abs(float(np.linalg.norm(imu_metrics.accel_mean)) - thresholds.gravity_magnitude)
        imu_stationary = (
            gyro_mean_norm <= thresholds.max_gyro_mean_norm
            and accel_mean_norm_error <= thresholds.max_accel_mean_norm_error
        )
        if not imu_stationary:
            self.stationary_frame_count = 0
            self.logger.trace(
                f"[EZUPT]: visual zero velocity rejected by IMU: "
                f"gyro_mean_norm={gyro_mean_norm}, accel_mean_norm_error={accel_mean_norm_error}"
            )

            return ZeroVelocityTrackerState.NON_ZERO_VELOCITY

        self.stationary_frame_count += 1
        return (
            ZeroVelocityTrackerState.ZERO_VELOCITY
            if self.stationary_frame_count >= thresholds.min_stationary_frames
            else ZeroVelocityTrackerState.NON_ZERO_VELOCITY
        )
