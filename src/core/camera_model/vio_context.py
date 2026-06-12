from dataclasses import dataclass, field

import gtsam
import numpy as np

from core.camera_model.stereo_camera_ctx import StereoContext
from dataset.sensor_config import IMUSensor


@dataclass(frozen=True)
class ImuContext:
    """IMU context."""

    frequency: float  # [Hz]

    accel_noise_destiny: float  # [m / s^2 / sqrt(Hz)]
    gyro_noise_destiny: float  # [rad / s / sqrt(Hz)]

    accel_random_walk: float  # [m / s ^ 3 / sqrt(Hz)]
    gyro_random_walk: float  # [rad / s ^ 2 / sqrt(Hz)]

    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))  # [m / s^2]

    def pim_params(self) -> gtsam.PreintegrationParams:
        """Get the PIM parameters."""
        params = gtsam.PreintegrationParams(self.gravity)
        params.setAccelerometerCovariance(np.eye(3) * (self.accel_noise_destiny**2))
        params.setGyroscopeCovariance(np.eye(3) * (self.gyro_noise_destiny**2))
        # GTSAM expects a 3x3 integration covariance here. Passing a 1x1 matrix
        # can corrupt the preintegration covariance and later produce NaNs in the
        # IMU factor noise model.
        params.setIntegrationCovariance(np.eye(3) * (1e-8 * 3))
        params.setUse2ndOrderCoriolis(False)
        return params

    @classmethod
    def from_imu_config(cls, imu_config: IMUSensor) -> "ImuContext":
        """Create an IMU context from an IMU configuration."""
        return cls(
            frequency=imu_config["rate_hz"],
            accel_noise_destiny=imu_config["accelerometer_noise_density"],
            gyro_noise_destiny=imu_config["gyroscope_noise_density"],
            accel_random_walk=imu_config["accelerometer_random_walk"],
            gyro_random_walk=imu_config["gyroscope_random_walk"],
        )

    @classmethod
    def empty(cls) -> "ImuContext":
        """Create an empty IMU context."""
        return cls(
            frequency=100.0,
            accel_noise_destiny=0.01,
            gyro_noise_destiny=0.01,
            accel_random_walk=3.0000e-3,
            gyro_random_walk=1.9393e-05,
        )


@dataclass(frozen=True)
class VioContext:
    """VIO context."""

    stereo: StereoContext
    imu: ImuContext

    @classmethod
    def from_stereo_and_imu_config(cls, stereo_ctx: StereoContext, imu_ctx: ImuContext) -> "VioContext":
        """Create a VIO context from a stereo and IMU configuration."""
        return cls(
            stereo=stereo_ctx,
            imu=imu_ctx,
        )
