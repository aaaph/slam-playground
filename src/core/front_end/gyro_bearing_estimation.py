from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.spatial.transform import Rotation


@dataclass(frozen=True, slots=True)
class GyroDelta:
    """Preintegrated gyro rotation delta linearized around current gyro bias."""

    rotation: Rotation
    bias_jacobian: NDArray[np.float64]
    dt_sec: float


class GyroBearingEstimation:
    """Collect bearing observations for gyro-bias estimation."""

    def __init__(self, observation_capacity: int = 4096, frame_capacity: int = 256) -> None:
        """Initialize the local arena buffer."""
        self._arena = _GyroBearingArenaBuffer(observation_capacity, frame_capacity)

    def add_observations(
        self,
        frame_id: int,
        timestamp_ns: float,
        feat_ids: NDArray[np.int32],
        left_bearings: NDArray[np.float64],
        gyro_delta: GyroDelta,
    ) -> slice:
        """Append one frame of bearing observations and return the observation range."""
        return self._arena.add_observations(frame_id, timestamp_ns, feat_ids, left_bearings, gyro_delta)


class _GyroBearingArenaBuffer:
    """Local SoA arena for mixed-type bearing estimation columns."""

    def __init__(self, observation_capacity: int, frame_capacity: int) -> None:
        self.frame_size = 0
        self.observation_size = 0

        self.frame_ids = np.empty(frame_capacity, dtype=np.int64)
        self.timestamps_ns = np.empty(frame_capacity, dtype=np.float64)
        self.gyro_delta_rotvecs = np.empty((frame_capacity, 3), dtype=np.float64)
        self.gyro_delta_bias_jacobians = np.empty((frame_capacity, 3, 3), dtype=np.float64)
        self.gyro_delta_dt_sec = np.empty(frame_capacity, dtype=np.float64)

        self.observation_frame_slots = np.empty(observation_capacity, dtype=np.int32)
        self.feat_ids = np.empty(observation_capacity, dtype=np.int32)
        self.left_bearings = np.empty((observation_capacity, 3), dtype=np.float64)

    def add_observations(
        self,
        frame_id: int,
        timestamp_ns: float,
        feat_ids: NDArray[np.int32],
        left_bearings: NDArray[np.float64],
        gyro_delta: GyroDelta,
    ) -> slice:
        if feat_ids.shape[0] > self.feat_ids.shape[0]:
            raise ValueError("Observation batch does not fit into the arena")
        frame_overflow = self.frame_size + 1 > self.frame_ids.shape[0]
        observation_overflow = self.observation_size + feat_ids.shape[0] > self.feat_ids.shape[0]
        if frame_overflow or observation_overflow:
            self.reset()

        frame_slot = self._append_frame(frame_id, timestamp_ns, gyro_delta)

        start = self.observation_size
        stop = start + feat_ids.shape[0]

        self.observation_frame_slots[start:stop] = frame_slot
        self.feat_ids[start:stop] = feat_ids
        self.left_bearings[start:stop] = left_bearings
        self.observation_size = stop
        return slice(start, stop)

    def reset(self) -> None:
        self.frame_size = 0
        self.observation_size = 0

    def _append_frame(self, frame_id: int, timestamp_ns: float, gyro_delta: GyroDelta) -> int:
        frame_slot = self.frame_size
        self.frame_ids[frame_slot] = frame_id
        self.timestamps_ns[frame_slot] = timestamp_ns
        self.gyro_delta_rotvecs[frame_slot] = gyro_delta.rotation.as_rotvec()
        self.gyro_delta_bias_jacobians[frame_slot] = gyro_delta.bias_jacobian
        self.gyro_delta_dt_sec[frame_slot] = gyro_delta.dt_sec
        self.frame_size += 1
        return frame_slot
