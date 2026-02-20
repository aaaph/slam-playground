from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Timestamp = float
Point2 = tuple[float, float]
Matrix = NDArray[np.float32]  # shape: (3, 3)
Vector = NDArray[np.float32]  # shape: (3,)


class FeatureLifecycle(Enum):
    """Feature lifecycle."""

    ACTIVE = 1
    LOST = 2


class FeatureStatus(Enum):
    """Feature status."""

    NEW = 0
    TRACKED = 1
    LOST = 2
    STABLE = 3
    UNSTABLE = 4


@dataclass(slots=True)
class Measurement:
    """Measurement of a feature."""

    timestamp: Timestamp
    left: Point2
    right: Point2 | None = None

    def is_stereo(self) -> bool:
        """Check if the measurement is a stereo measurement."""
        return self.right is not None

    def is_left_only(self) -> bool:
        """Check if the measurement is a left only measurement."""
        return self.right is None

    def as_tuple(self) -> tuple[Timestamp, Point2, Point2 | None]:
        """Convert the measurement to a tuple."""
        return (self.timestamp, self.left, self.right)

    def pair(self) -> tuple[Point2, Point2]:
        """Get the pair of the measurement."""
        return (self.left, self.right)


class Feature:
    """Represents a tracked feature with associated points and linear system matrices."""

    def __init__(self, feat_id: int, capacity: int = 4, spawned_timestamp: float = -1.0) -> None:
        """Initialize a feature with the given ID."""
        self.feat_id = feat_id
        self.capacity = capacity
        self.size = 0
        self.head = 0
        self.iteration_life = 0

        self.A = np.zeros((3, 3), dtype=np.float32)
        self.b = np.zeros(3, dtype=np.float32)

        self.p_fw = None
        self.valid = False
        self.state = FeatureStatus.NEW

        self.ts = np.full(self.capacity, -1.0, np.int64)
        self.cam_id = np.full(self.capacity, -1, np.int32)
        self.u = np.full(self.capacity, np.nan, np.float32)
        self.v = np.full(self.capacity, np.nan, np.float32)

        self.active_timestamp = -1.0
        self.left_pair_idx: None | int = None
        self.right_pair_idx: None | int = None

        self.iteration_life_threshold = 0
        self.spawned_timestamp = spawned_timestamp
        self.max_cond_a = 10000.0
        self.max_depth = 60.0
        self.min_depth = 0.15

    def _add(self, ts: Timestamp, cam_id: Literal[0, 1], uv: Point2) -> int:
        """Add a new observation to the feature."""
        index = self.head
        u, v = uv
        self.ts[index] = ts
        self.cam_id[index] = cam_id
        self.u[index] = u
        self.v[index] = v
        self.head = (self.head + 1) % self.capacity
        self.size = np.minimum(self.size + 1, self.capacity)
        self.active_timestamp = max(self.active_timestamp, ts)
        if self.size > 2 and self.state == FeatureStatus.NEW:  # noqa: PLR2004
            self.state = FeatureStatus.TRACKED

        return index

    def unstable(self) -> None:
        """Mark the feature as unstable."""
        self.state = FeatureStatus.UNSTABLE

    def apply_stereo_pair(self, ts: Timestamp, left_uv: Point2, right_uv: Point2) -> None:
        """Apply a stereo pair to the feature."""
        left_idx = self._add(ts, 0, left_uv)
        right_idx = self._add(ts, 1, right_uv)
        self.active_timestamp = max(self.active_timestamp, ts)
        self.left_pair_idx = left_idx
        self.right_pair_idx = right_idx
        self.iteration_life += 1

    def apply_left_only(self, ts: Timestamp, left_uv: Point2) -> None:
        """Apply a left point to the feature."""
        left_idx = self._add(ts, 0, left_uv)
        self.active_timestamp = max(self.active_timestamp, ts)
        self.left_pair_idx = left_idx
        self.right_pair_idx = None
        self.iteration_life += 1

    def apply_linear_system_update(self, delta_a: Matrix, delta_b: Vector) -> None:
        """Apply a linear system update to the feature."""
        self.A += delta_a
        self.b += delta_b

    def get_uv_by_timestamp(self, ts: Timestamp) -> list[tuple[Literal[0, 1], float, float]]:
        """Get the uv by timestamp. Method could return 1 point or 2 per 1 timestamp."""
        result: list[tuple[Literal[0, 1], float, float]] = []
        mask = self.ts == ts
        for cam_id, u, v in zip(self.cam_id[mask], self.u[mask], self.v[mask], strict=True):
            result.append((cam_id, u, v))
        return result

    def get_active_stereo_pair(self) -> tuple[Timestamp, Point2, Point2 | None]:
        """Get the active stereo pair of the feature."""
        if self.size < 1 or self.left_pair_idx is None:
            msg = f"Feature has no active left point, feat_id: {self.feat_id}"
            raise ValueError(msg)

        left_idx = self.left_pair_idx
        timestamp = self.ts[left_idx]
        left_pt = (self.u[left_idx], self.v[left_idx])

        right_pt = None
        if self.right_pair_idx is not None:
            r_idx = self.right_pair_idx
            right_pt = (self.u[r_idx], self.v[r_idx])
        return timestamp, left_pt, right_pt

    def get_active_measurement(self) -> Measurement:
        """Get the active measurement of the feature."""
        timestamp, left_uv, right_uv = self.get_active_stereo_pair()
        return Measurement(timestamp=timestamp, left=left_uv, right=right_uv)

    def __repr__(self) -> str:
        """Return the representation of the feature."""
        # rows = []
        state = self.state
        size = self.size
        head = self.head
        """ for i in range(self.size):
            rows.append(f"ts: {self.ts[i]}, cam_id: {self.cam_id[i]}, u: {self.u[i]}, v: {self.v[i]}") """
        return f"Feature(feat_id={self.feat_id}, state: {state}, size: {size}, head: {head})"

    def obs_count(self) -> int:
        """Get the number of observations for the feature."""
        return self.size

    def feature_color(self) -> tuple[int, int, int]:
        """Get the color of the feature."""
        match self.state:
            case FeatureStatus.NEW:
                return (0, 255, 0)  # green
            case FeatureStatus.TRACKED:
                return (255, 0, 0)  # blue
            case FeatureStatus.LOST:
                return (128, 128, 128)  # grey
            case FeatureStatus.STABLE:
                return (255, 0, 255)  # purple
            case FeatureStatus.UNSTABLE:
                return (0, 0, 255)  # red
            case _:
                return (255, 0, 0)

    @property
    def debug_color(self) -> tuple[int, int, int]:
        """Get the debug color of the feature."""
        return (0, 0, 255)

    def get_tail(self, cam_id: Literal[0, 1]) -> list[Point2]:
        """Get the tail of the feature."""
        if self.size < 1:
            raise ValueError("Feature has no observations")
        camera_mask = self.cam_id == cam_id
        timestamp_mask = self.ts != self.active_timestamp
        mask = camera_mask & timestamp_mask
        return list(zip(self.u[mask], self.v[mask], strict=False))

    def iterate(self) -> Iterator[tuple[float, Literal[0, 1], float, float]]:
        """Iterate over the feature."""
        index = 0
        while index < self.size:
            yield self.ts[index], self.cam_id[index], self.u[index], self.v[index]
            index += 1

    @property
    def ready_to_triangulate(self) -> bool:
        """Check if the feature is ready to be triangulated."""
        return self.iteration_life > self.iteration_life_threshold and self.state == FeatureStatus.TRACKED

    @staticmethod
    def spawn_from_left_and_right(
        feat_id: int, ts: float, left_uv: Point2, right_uv: Point2, feat_capacity: int = 4
    ) -> "Feature":
        """Spawn a feature from a left and right observation."""
        feature = Feature(feat_id, capacity=feat_capacity, spawned_timestamp=ts)
        feature.apply_stereo_pair(ts, left_uv, right_uv)
        return feature

    @classmethod
    def spawn_from_ndarray(cls, ndarray: NDArray[np.float32]) -> "Feature":
        """
        Spawn a feature from a ndarray.

        ndarray: [feat_id, ts, left_u, left_v, right_u, right_v, status]
        feat_id: int
        ts: float
        left_uv: Point2
        right_uv: Point2 | None
        status: FeatureStatus
        """
        feat_id = int(ndarray[0])
        ts = float(ndarray[1])
        left_uv = (float(ndarray[2]), float(ndarray[3]))
        right_uv = None if np.isnan(ndarray[4]) else (float(ndarray[4]), float(ndarray[5]))
        status = FeatureStatus(int(ndarray[6]))
        feature = cls(feat_id, capacity=4, spawned_timestamp=ts)
        if right_uv is not None:
            feature.apply_stereo_pair(ts, left_uv, right_uv)
        else:
            feature.apply_left_only(ts, left_uv)
        feature.state = status
        return feature
