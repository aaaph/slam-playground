from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

import gtsam
from dataset.euroc import GroundTruth

gyro_sigma = 1.6968e-04
accel_sigma = 2.0000e-3
gyro_walk = 1.9393e-05
accel_walk = 3.0000e-3


@dataclass(slots=True)
class InertialIntegrationState:
    """Optional initial state for inertial integration."""

    gravity: np.ndarray | None = None
    accel_bias: np.ndarray | None = None
    gyro_bias: np.ndarray | None = None
    position_vector: np.ndarray | None = None
    velocity_vector: np.ndarray | None = None
    rotation_matrix: np.ndarray | None = None


class ImuInitializer:
    """Imu initializer."""

    def __init__(self, capacity: int) -> None:
        """Initialize the imu initializer."""
        self.capacity = capacity
        self.buffer = np.full((self.capacity, 7), np.nan)
        self.idx = 0

    def add_batch(self, accel_batch: np.ndarray, gyro_batch: np.ndarray, timestamp_batch: np.ndarray) -> None:
        """Add a batch of imu measurements to the buffer."""
        batch_size = accel_batch.shape[0]
        if batch_size == 0:
            return
        if self.idx + batch_size > self.capacity:
            msg = f"Batch size {self.idx + batch_size} is greater than capacity {self.capacity}"
            raise ValueError(msg)
        batch = np.column_stack((timestamp_batch, accel_batch, gyro_batch))
        self.buffer[self.idx : self.idx + batch_size, :] = batch
        self.idx = (self.idx + batch_size) % self.capacity

    def create_initial_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create the initial bias from the buffer."""
        accel_batch = self.buffer[~np.isnan(self.buffer[:, 1]), 1:4]
        gyro_batch = self.buffer[~np.isnan(self.buffer[:, 4]), 4:7]

        if accel_batch.size == 0 or gyro_batch.size == 0:
            return np.zeros(3), np.zeros(3), Rotation.identity().as_quat()

        gyro_bias = np.mean(gyro_batch, axis=0)
        accel_mean = np.mean(accel_batch, axis=0)
        accel_norm = np.linalg.norm(accel_mean)
        if accel_norm == 0:
            return gyro_bias, np.zeros(3), Rotation.identity().as_quat()

        z_axis = accel_mean / accel_norm
        x_axis = np.array([1, 0, 0])
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_axis_norm = np.linalg.norm(x_axis)
        if x_axis_norm == 0:
            return gyro_bias, np.zeros(3), Rotation.identity().as_quat()
        x_axis /= x_axis_norm
        y_axis = np.cross(z_axis, x_axis)

        rot = Rotation.from_matrix(np.column_stack((x_axis, y_axis, z_axis)))
        accel_bias = np.zeros(3)
        return gyro_bias, accel_bias, rot.as_quat()


class InertialIntegration:
    """Inertial integration."""

    def __init__(
        self,
        timestamp: float,
        initial_state: InertialIntegrationState | None = None,
    ) -> None:
        """Initialize the inertial integration."""
        initial_state = initial_state or InertialIntegrationState()
        self.timestamp = timestamp
        self.gravity = (
            np.asarray(initial_state.gravity) if initial_state.gravity is not None else np.array([0, 0, 9.81])
        )
        params = gtsam.PreintegrationParams(self.gravity)
        params.setAccelerometerCovariance(np.eye(3) * (accel_sigma**2))
        params.setGyroscopeCovariance(np.eye(3) * (gyro_sigma**2))
        params.setIntegrationCovariance(np.diag([1e-4 * 3]))
        # params.setUse2ndOrderCoriolis(True)
        # params.setOmegaCoriolis(np.array([0, 0, 0]))
        # params.setBodyPSensor(gtsam.Pose3())
        self.params = params
        self.current_bias = gtsam.imuBias.ConstantBias(
            np.asarray(initial_state.accel_bias) if initial_state.accel_bias is not None else np.zeros(3),
            np.asarray(initial_state.gyro_bias) if initial_state.gyro_bias is not None else np.zeros(3),
        )
        self.pim = gtsam.PreintegratedImuMeasurements(self.params, self.current_bias)
        self.pim.resetIntegrationAndSetBias(self.current_bias)

        rot = gtsam.Rot3(
            np.asarray(initial_state.rotation_matrix) if initial_state.rotation_matrix is not None else np.eye(3)
        )
        vec = gtsam.Point3(
            *(
                np.asarray(initial_state.position_vector)
                if initial_state.position_vector is not None
                else np.zeros(3)
            )
        )
        pose = gtsam.Pose3(rot, vec)
        self.nav_state = gtsam.NavState(
            pose=pose,
            v=(
                np.asarray(initial_state.velocity_vector)
                if initial_state.velocity_vector is not None
                else np.zeros(3)
            ),
        )
        self.last_keyframe_nav_state = gtsam.NavState(
            pose=pose,
            v=(
                np.asarray(initial_state.velocity_vector)
                if initial_state.velocity_vector is not None
                else np.zeros(3)
            ),
        )
        self.imu_initializer = ImuInitializer(capacity=1000)
        self.init = False

    @classmethod
    def from_ground_truth(cls, gravity: np.ndarray, ground_truth: GroundTruth) -> "InertialIntegration":
        """Initialize the inertial integration from a ground truth."""
        timestamp = ground_truth["timestamp"]
        accel_bias = ground_truth["gt_acc_bias"]
        gyro_bias = ground_truth["gt_gyro_bias"]
        # position_vector = ground_truth["gt_position"]
        # rotation_matrix = ground_truth["gt_orientation"]
        return cls(
            timestamp=timestamp,
            initial_state=InertialIntegrationState(
                gravity=gravity,
                accel_bias=np.array(accel_bias),
                gyro_bias=np.array(gyro_bias),
                position_vector=np.array(np.zeros(3)),
                velocity_vector=np.zeros(3),
                rotation_matrix=Rotation.from_quat(np.array([0, 0, 0, 1])).as_matrix(),
            ),
        )

    def _integrate(self, accel: np.ndarray, gyro: np.ndarray, dt: float) -> None:
        # print(f"Integrating measurement with accel {accel} and gyro {gyro} and dt {dt}")
        self.pim.integrateMeasurement(accel, gyro, dt)

    def integrate_batch(self, accel_batch: np.ndarray, gyro_batch: np.ndarray, timestamp_batch: np.ndarray) -> int:
        """
        Integrate a batch of inertial measurements.

        accel_batch, gyro_batch and timestamp_batch should be same size.
        """
        past_ts_mask = timestamp_batch < self.timestamp
        if np.all(past_ts_mask):
            # all timestamps in the past -> add into initializer
            self.imu_initializer.add_batch(accel_batch, gyro_batch, timestamp_batch)
            return 0
        if not self.init:
            _ = self.imu_initializer.create_initial_state()
            # print(f"initial_state: {initial_state}")
            # need to update the bias + rotation matrix
            self.init = True
        future_ts_mask = timestamp_batch > self.timestamp
        accel_filtered = accel_batch[future_ts_mask]
        gyro_filtered = gyro_batch[future_ts_mask]
        timestamp_filtered = timestamp_batch[future_ts_mask]
        ts_size = timestamp_filtered.shape[0]
        if ts_size == 0:
            return 0

        timestamp_offset = np.empty(ts_size)
        timestamp_offset[0] = self.timestamp
        timestamp_offset[1:] = timestamp_filtered[:-1]
        dt_batch = (timestamp_filtered - timestamp_offset) / 1e9
        valid_dt_mask = dt_batch > 0
        integrated_count = 0
        for accel, gyro, dt in zip(
            accel_filtered[valid_dt_mask], gyro_filtered[valid_dt_mask], dt_batch[valid_dt_mask], strict=True
        ):
            self._integrate(accel=accel, gyro=gyro, dt=dt)
            integrated_count += 1
        if integrated_count > 0:
            self.timestamp = timestamp_batch[-1]
        return integrated_count

    def integrate_and_predict(self, accel: np.ndarray, gyro: np.ndarray, timestamp: np.ndarray) -> gtsam.NavState:
        """Predict the state using the preintegrated measurements."""
        count = self.integrate_batch(accel, gyro, timestamp)
        if count > 0:
            current_best_bias = self.current_bias
            # print(f"current_best_bias: {current_best_bias}")
            prediction = self.pim.predict(self.last_keyframe_nav_state, current_best_bias)
            self.nav_state = prediction
        return self.nav_state
