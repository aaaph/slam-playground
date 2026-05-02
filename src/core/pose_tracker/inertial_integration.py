from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

gyro_sigma = 1.6968e-04
accel_sigma = 2.0000e-3
gyro_walk = 1.9393e-05
accel_walk = 3.0000e-3


@dataclass(slots=True, frozen=True)
class ImuBufferInfo:
    """Info of the imu buffer."""

    size: int
    reset_ts: float | None
    first_buffer_ts: float | None
    last_buffer_ts: float | None


@dataclass(slots=True, frozen=True)
class InitialState:
    """Initial state for inertial integration."""

    gyro_bias: np.ndarray
    accel_bias: np.ndarray
    rotation: Rotation
    gyro_std: np.ndarray
    accel_std: np.ndarray
    gyro_mean: np.ndarray
    accel_mean: np.ndarray

    @classmethod
    def empty(cls) -> "InitialState":
        """Create an empty initial state."""
        return cls(
            gyro_bias=np.zeros(3),
            accel_bias=np.zeros(3),
            rotation=Rotation.identity(),
            gyro_std=np.zeros(3),
            accel_std=np.zeros(3),
            gyro_mean=np.zeros(3),
            accel_mean=np.zeros(3),
        )

    @classmethod
    def from_gyro_bias(cls, gyro_bias: np.ndarray) -> "InitialState":
        """Create an initial state from a gyro bias."""
        return cls(
            gyro_bias=gyro_bias,
            accel_bias=np.zeros(3),
            rotation=Rotation.identity(),
            gyro_std=np.zeros(3),
            accel_std=np.zeros(3),
            gyro_mean=np.zeros(3),
            accel_mean=np.zeros(3),
        )

    def __repr__(self) -> str:
        """Return a string representation of the initial state."""
        quat = self.rotation.as_quat()
        return (
            "InitialState("
            f"gyro_bias={self.gyro_bias}, accel_bias={self.accel_bias}, "
            f"quat={quat}, gyro_std={self.gyro_std}, accel_std={self.accel_std}, "
            f"gyro_mean={self.gyro_mean}, accel_mean={self.accel_mean})"
        )


@dataclass(slots=True)
class InertialIntegrationState:
    """Optional initial state for inertial integration."""

    gravity: np.ndarray | None = None
    accel_bias: np.ndarray | None = None
    gyro_bias: np.ndarray | None = None
    position_vector: np.ndarray | None = None
    velocity_vector: np.ndarray | None = None
    rotation_matrix: np.ndarray | None = None


class ImuBuffer:
    """Imu buffer."""

    def __init__(self, capacity: int) -> None:
        """Initialize the imu buffer."""
        self.capacity = capacity
        self.buffer = np.full((self.capacity, 8), np.nan)
        self.idx = 0
        self.size = 0
        self.reset_ts: None | float = None
        self.last_batch_slice: tuple[int, int] = (0, 0)

    def add_batch(self, accel_batch: np.ndarray, gyro_batch: np.ndarray, timestamp_batch: np.ndarray) -> None:
        """Add a batch of imu measurements to the buffer."""
        batch_size = accel_batch.shape[0]
        if batch_size == 0:
            return
        if self.idx + batch_size > self.capacity:
            msg = f"Batch size {self.idx + batch_size} is greater than capacity {self.capacity}"
            raise ValueError(msg)
        dt_batch = np.zeros(batch_size, dtype=np.float32)
        if self.reset_ts is not None:
            dt_batch[0] = (timestamp_batch[0] - self.reset_ts) * 1e-9
        if batch_size > 1:
            dt_batch[1:] = (timestamp_batch[1:] - timestamp_batch[:-1]) * 1e-9
        batch = np.column_stack((timestamp_batch, accel_batch, gyro_batch, dt_batch))

        self.buffer[self.idx : self.idx + batch_size, :] = batch
        self.last_batch_slice = (self.idx, self.idx + batch_size)
        self.idx += batch_size
        self.size += batch_size
        self.reset_ts = timestamp_batch[-1]

    def reset(self, timestamp: float) -> None:
        """Reset the imu buffer and set timestamp to the last timestamp in the buffer."""
        self.reset_ts = timestamp
        self.idx = 0
        self.size = 0
        self.buffer.fill(np.nan)
        self.last_batch_slice = (0, 0)

    def iterate_last_batch(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Iterate over the last batch of imu measurements."""
        start, end = self.last_batch_slice
        batch_view = self.buffer[start:end, :]
        for i in range(batch_view.shape[0]):
            dt = batch_view[i, 7]
            if dt <= 0:
                continue
            accel = batch_view[i, 1:4]
            gyro = batch_view[i, 4:7]
            yield accel, gyro, dt

    def iterate_full_buffer(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Iterate over all the imu measurements in the buffer."""
        for i in range(self.size):
            dt = self.buffer[i, 7]
            if dt <= 0:
                continue
            accel = self.buffer[i, 1:4]
            gyro = self.buffer[i, 4:7]
            yield accel, gyro, dt

    def info(self) -> ImuBufferInfo:
        """Get the info of the imu buffer."""
        if self.size == 0:
            return ImuBufferInfo(
                size=0, reset_ts=self.reset_ts, first_buffer_ts=self.reset_ts, last_buffer_ts=self.reset_ts
            )
        reset_ts = self.reset_ts
        first_buffer_ts = self.buffer[0, 0]
        last_buffer_ts = self.buffer[self.size - 1, 0]
        return ImuBufferInfo(
            size=self.size, reset_ts=reset_ts, first_buffer_ts=first_buffer_ts, last_buffer_ts=last_buffer_ts
        )

    def get_batch(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the batch of imu measurements from the buffer."""
        accel_batch = self.buffer[: self.size, 1:4]
        gyro_batch = self.buffer[: self.size, 4:7]
        # timestamp_batch = self.buffer[: self.size, 0]
        return accel_batch, gyro_batch

    def get_gyro(self) -> np.ndarray:
        """Get the gyro measurements from the buffer."""
        return self.buffer[: self.size, 4:7]

    def get_accel(self) -> np.ndarray:
        """Get the accel measurements from the buffer."""
        return self.buffer[: self.size, 1:4]

    def get_6d(self) -> np.ndarray:
        """Get the 6d measurements from the buffer."""
        return self.buffer[: self.size, 1:7]

    def get_timestamp(self) -> np.ndarray:
        """Get the timestamp measurements from the buffer."""
        return self.buffer[: self.size, 0]

    def create_initial_state(self) -> InitialState:
        """Create the initial bias from the buffer."""
        accel_batch, gyro_batch = self.get_batch()

        if accel_batch.size == 0 or gyro_batch.size == 0:
            return InitialState.empty()

        gyro_bias = np.mean(gyro_batch, axis=0)
        accel_mean = np.mean(accel_batch, axis=0)
        accel_norm = np.linalg.norm(accel_mean)
        if accel_norm == 0:
            return InitialState.empty()
        # Gram-Schmidt
        z_axis = accel_mean / accel_norm
        x_axis = np.array([1, 0, 0])
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_axis_norm = np.linalg.norm(x_axis)
        if x_axis_norm == 0:
            return InitialState.from_gyro_bias(gyro_bias)
        x_axis /= x_axis_norm
        y_axis = np.cross(z_axis, x_axis)
        rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
        rot_wb = Rotation.from_matrix(rot_matrix.T)
        accel_bias = np.zeros(3)
        return InitialState(
            gyro_bias=gyro_bias,
            accel_bias=accel_bias,
            rotation=rot_wb,
            gyro_std=np.std(gyro_batch, axis=0),
            accel_std=np.std(accel_batch, axis=0),
            gyro_mean=gyro_bias,
            accel_mean=accel_mean,
        )

    @classmethod
    def from_batch(
        cls, accel_batch: np.ndarray, gyro_batch: np.ndarray, timestamp_batch: np.ndarray
    ) -> "ImuBuffer":
        """Create an ImuBuffer from a full batch."""
        all_same_shape = accel_batch.shape[0] == gyro_batch.shape[0] == timestamp_batch.shape[0]
        if not all_same_shape:
            msg = "accel_batch, gyro_batch and timestamp_batch must have the same shape"
            raise ValueError(msg)
        buffer = cls(capacity=accel_batch.shape[0])
        buffer.add_batch(accel_batch, gyro_batch, timestamp_batch)
        return buffer
