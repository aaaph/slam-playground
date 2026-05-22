from collections import OrderedDict
from typing import Self

import numpy as np
from numpy.typing import NDArray
from zmq import IntEnum

FeatureId = int
Vector3d = NDArray[np.float32]


class LocalMapSchema(IntEnum):
    """Local map schema."""

    FEAT_ID = 0
    X = 1
    Y = 2
    Z = 3
    HEALTH = 4
    LAST_TS = 5


class LocalMap:
    """Local map backed by a fixed-size table."""

    def __init__(self, capacity: int = 1000) -> None:
        """Initialize the local map."""
        self.capacity = capacity
        self.stable_health_threshold = -3
        # Public attribute is kept for compatibility with the current tests and LRU checks.
        self.landmarks: OrderedDict[FeatureId, int] = OrderedDict()
        # feat_id, x, y, z, health, last_ts
        self._data = np.full((capacity, LocalMapSchema.LAST_TS + 1), np.nan, dtype=np.float32)
        self._feat_id_to_idx: dict[FeatureId, int] = {}
        self._free_slots = list(range(capacity - 1, -1, -1))

    @classmethod
    def from_capacity(cls, capacity: int) -> Self:
        """Create a local map from a capacity."""
        return cls(capacity)

    def _get_free_slot(self) -> int:
        """Get a free slot or evict the least recently used point."""
        if self._free_slots:
            return self._free_slots.pop()

        feat_id, idx = self.landmarks.popitem(last=False)
        self._feat_id_to_idx.pop(feat_id, None)
        self._data[idx].fill(np.nan)
        return idx

    def _get_existing_slot(self, feat_id: FeatureId) -> int:
        """Get an existing slot for the feature id."""
        idx = self._feat_id_to_idx.get(feat_id)
        if idx is None:
            msg = f"Feature with ID {feat_id} not found"
            raise ValueError(msg)
        return idx

    def _find_slots(self, feat_ids: NDArray[np.int32]) -> NDArray[np.int32]:
        """Find the slots of the features in the local map."""
        if len(feat_ids) != len(np.unique(feat_ids)):
            raise ValueError("Duplicate feature IDs")
        output = np.zeros_like(feat_ids, dtype=np.int32)
        for i, feat_id in enumerate(feat_ids):
            feat_id_val = feat_id.item()
            idx = self._feat_id_to_idx.get(feat_id_val, None)
            if idx is None:
                msg = f"Feature with ID {feat_id_val} not found"
                raise ValueError(msg)
            output[i] = idx
        return output

    def add_point(self, feat_id: FeatureId, point_3d: Vector3d) -> None:
        """Add or update a point in the local map."""
        idx = self._feat_id_to_idx.get(feat_id)
        if idx is None:
            idx = self._get_free_slot()
            self._feat_id_to_idx[feat_id] = idx

        self._data[idx, LocalMapSchema.FEAT_ID] = feat_id
        self._data[idx, LocalMapSchema.X : LocalMapSchema.Z + 1] = point_3d
        if np.isnan(self._data[idx, LocalMapSchema.HEALTH]):
            self._data[idx, LocalMapSchema.HEALTH] = 1.0
        if np.isnan(self._data[idx, LocalMapSchema.LAST_TS]):
            self._data[idx, LocalMapSchema.LAST_TS] = 0.0

        self.landmarks[feat_id] = idx
        self.landmarks.move_to_end(feat_id)

    def add_points(self, new_points: dict[FeatureId, Vector3d]) -> None:
        """Add points to the local map."""
        for feat_id, point_3d in new_points.items():
            self.add_point(feat_id, point_3d)

    def get_point(self, feat_id: FeatureId) -> Vector3d | None:
        """Get a point from the local map."""
        idx = self._feat_id_to_idx.get(feat_id)
        if idx is None:
            return None

        self.landmarks.move_to_end(feat_id)
        return self._data[idx, LocalMapSchema.X : LocalMapSchema.Z + 1].copy()

    def increase_health(self, feat_ids: NDArray[np.int32]) -> NDArray[np.int32]:
        """Increase the health score for an existing landmark."""
        if feat_ids.size == 0:
            return np.empty(0, dtype=np.int32)
        indexes = self._find_slots(feat_ids)
        self._data[indexes, LocalMapSchema.HEALTH] += 1.0
        for feat_id in feat_ids:
            self.landmarks.move_to_end(int(feat_id))
        return self._data[indexes, LocalMapSchema.HEALTH].astype(np.int32, copy=False)

    def decrease_health(self, feat_ids: NDArray[np.int32]) -> NDArray[np.int32]:
        """Decrease the health score for an existing landmark."""
        if feat_ids.size == 0:
            return np.empty(0, dtype=np.int32)
        indexes = self._find_slots(feat_ids)
        self._data[indexes, LocalMapSchema.HEALTH] -= 1.0
        for feat_id in feat_ids:
            self.landmarks.move_to_end(int(feat_id))
        return self._data[indexes, LocalMapSchema.HEALTH].astype(np.int32, copy=False)

    def get_batch(self, feat_ids: NDArray[np.int32]) -> tuple[NDArray[np.bool_], NDArray[np.float32]]:
        """Get a batch of rows aligned to input feat ids as [feat_id, x, y, z]."""
        mask = np.zeros(feat_ids.shape[0], dtype=bool)
        points = np.full((feat_ids.shape[0], LocalMapSchema.Z + 1), np.nan, dtype=np.float32)
        points[:, LocalMapSchema.FEAT_ID] = feat_ids.astype(np.float32, copy=False)

        for i, feat_id in enumerate(feat_ids):
            idx = self._feat_id_to_idx.get(int(feat_id))
            if idx is None:
                continue
            mask[i] = True
            points[i, LocalMapSchema.X : LocalMapSchema.Z + 1] = self._data[
                idx, LocalMapSchema.X : LocalMapSchema.Z + 1
            ]

        return mask, points

    def get_stable_batch(self, feat_ids: NDArray[np.int32]) -> tuple[NDArray[np.bool_], NDArray[np.float32]]:
        """Get stable rows aligned to input feat ids as [feat_id, x, y, z, health, last_ts]."""
        mask = np.zeros(feat_ids.shape[0], dtype=bool)
        points = np.full((feat_ids.shape[0], LocalMapSchema.LAST_TS + 1), np.nan, dtype=np.float32)
        points[:, LocalMapSchema.FEAT_ID] = feat_ids.astype(np.float32, copy=False)

        for i, feat_id in enumerate(feat_ids):
            idx = self._feat_id_to_idx.get(int(feat_id))
            if idx is None:
                continue
            health = self._data[idx, LocalMapSchema.HEALTH]
            if health < self.stable_health_threshold:
                continue
            mask[i] = True
            points[i] = self._data[idx]

        return mask, points

    def exists(self, feat_id: FeatureId) -> bool:
        """Check if a point exists in the local map."""
        return feat_id in self._feat_id_to_idx

    def empty(self) -> bool:
        """Check if the local map is empty."""
        return len(self._feat_id_to_idx) == 0

    def clear(self) -> None:
        """Clear the local map."""
        self.landmarks.clear()
        self._feat_id_to_idx.clear()
        self._data.fill(np.nan)
        self._free_slots = list(range(self.capacity - 1, -1, -1))

    def add_ndarray(self, ndarray: NDArray[np.float32]) -> None:
        """Add rows shaped as [feat_id, x, y, z, ...] to the local map."""
        if ndarray.shape[0] == 0:
            return
        if ndarray.shape[0] > self.capacity:
            raise ValueError("Too many points to add")

        for row in ndarray:
            feat_id = int(row[LocalMapSchema.FEAT_ID])
            point_3d = row[LocalMapSchema.X : LocalMapSchema.Z + 1]
            self.add_point(feat_id, point_3d)
